# Guiding Field Configuration.
# Expose useful guiding center quantities starting from a EM field
# i.e. gradient of B, A_dagger etc (see GuidingField struct)
import numpy as np
from emFields.AB_dBfields.AB_dBfield import ABdBGuidingCenter, AB_dB_FieldBuilder
from emFields.ABfields.emFieldFactory import EMFieldFactory


class FiniteDFromAB(AB_dB_FieldBuilder):

    def __init__(self, config):
        self.mu = config.mu
        self.hx = config.hx

        self.field = EMFieldFactory(config.emField, config)

    def B_grad(self, x):
        ret = np.zeros(3)
        for j in range(3):
            x0 = np.array(x)
            x1 = np.array(x)
            x0[j] -= self.hx
            x1[j] += self.hx
            B1 = self.field.B(x1)
            B0 = self.field.B(x0)
            ret[j] = 0.5*(np.linalg.norm(B1) - np.linalg.norm(B0)) / self.hx
        return ret

    def B_Hessian(self, x):
        ret = np.zeros([3, 3])
        for j in range(3):
            x0 = np.array(x)
            x1 = np.array(x)
            x0[j] -= self.hx
            x1[j] += self.hx
            ret[:, j] = 0.5*(self.B_grad(x1) - self.B_grad(x0)) / self.hx

        return ret

    def compute(self, z):
        x = z[:3]

        A = self.field.A(x)
        B = self.field.B(x)
        Bnorm = np.linalg.norm(B)
        b = B / Bnorm

        u = z[3]
        Adag = A + u*b

        # COMPUTE GRADIENT(B),GRADIENT(phi),JAC(A_dagger)
        B_grad = np.zeros(3)
        b_jac = np.zeros([3, 3])
        for j in range(3):
            x0 = np.array(x)
            x1 = np.array(x)
            x0[j] -= self.hx
            x1[j] += self.hx
            B0 = self.field.B(x0)
            B1 = self.field.B(x1)
            B1norm = np.linalg.norm(B1)
            B0norm = np.linalg.norm(B0)

            B_grad[j] = 0.5*(B1norm - B0norm) / self.hx
            b_jac[:, j] = 0.5*(B1/B1norm - B0/B0norm) / self.hx

        # COMPUTE B_dagger
        Bdag = B
        Bdag[0] += u*(b_jac[2, 1] - b_jac[1, 2])
        Bdag[1] += u*(b_jac[0, 2] - b_jac[2, 0])
        Bdag[2] += u*(b_jac[1, 0] - b_jac[0, 1])

        BHessian = self.B_Hessian(x)

        return ABdBGuidingCenter(A=A, Adag=Adag, B=B, Bgrad=B_grad, b=b, Bnorm=Bnorm, BHessian=BHessian, Bdag=Bdag)
