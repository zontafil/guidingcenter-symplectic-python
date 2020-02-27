# Guiding Field Configuration.
# Expose useful guiding center quantities starting from a EM field
# i.e. gradient of B, A_dagger etc (see GuidingField struct)
import numpy as np
from emFields.AB_dBfields.AB_dBfield import ABdBGuidingCenter, AB_dB_FieldBuilder
from emFields.ABfields.emFieldFactory import EMFieldFactory
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class FiniteDFromA(AB_dB_FieldBuilder):

    def __init__(self, config):
        self.hx = config.hx
        self.drawZoomOut = 1.3
        self.eps = 0.32
        self.R0 = config.R0
        self.k = 1.7

        self.field = EMFieldFactory(config.emField, config)

    def B_grad(self, x):
        ret = np.zeros(3)
        for j in range(3):
            x0 = np.array(x)
            x1 = np.array(x)
            x0[j] -= self.hx
            x1[j] += self.hx
            B1 = self.B(x1)
            B0 = self.B(x0)
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

    def B(self, z):
        # compute B from A
        x = z[:3]
        A_jac = np.zeros([3, 3])
        B = np.zeros(3)

        for j in range(3):
            x0 = np.array(x)
            x1 = np.array(x)
            x0[j] -= self.hx
            x1[j] += self.hx
            A_jac[:, j] = 0.5*(self.field.A(x1) - self.field.A(x0)) / self.hx
        B[0] = A_jac[2, 1] - A_jac[1, 2]
        B[1] = A_jac[0, 2] - A_jac[2, 0]
        B[2] = A_jac[1, 0] - A_jac[0, 1]
        return B

    def compute(self, z):
        x = z[:3]

        A = self.field.A(x)
        B = self.B(x)
        Bnorm = np.linalg.norm(B)
        b = B / Bnorm

        u = z[3]
        Adag = A + u*b

        # COMPUTE GRADIENT(B),GRADIENT(phi),JAC(A_dagger)
        B_grad = np.zeros(3)
        b_jac = np.zeros([3, 3])
        Adag_jac = np.zeros([3, 3])
        for j in range(3):
            x0 = np.array(x)
            x1 = np.array(x)
            x0[j] -= self.hx
            x1[j] += self.hx
            B0 = self.B(x0)
            B1 = self.B(x1)
            B1norm = np.linalg.norm(B1)
            B0norm = np.linalg.norm(B0)

            Adag_jac[:, j] = 0.5*(self.field.A(x1) + u*B1/B1norm - self.field.A(x0) - u*B0/B0norm) / self.hx
            B_grad[j] = 0.5*(B1norm - B0norm) / self.hx
            b_jac[:, j] = 0.5*(B1/B1norm - B0/B0norm) / self.hx

        # COMPUTE B_dagger
        Bdag = np.array(B)
        Bdag[0] += u*(b_jac[2, 1] - b_jac[1, 2])
        Bdag[1] += u*(b_jac[0, 2] - b_jac[2, 0])
        Bdag[2] += u*(b_jac[1, 0] - b_jac[0, 1])

        BHessian = self.B_Hessian(x)

        return ABdBGuidingCenter(A=A, Adag_jac=Adag_jac, Adag=Adag,
                                 B=B, Bgrad=B_grad, b=b, Bnorm=Bnorm, BHessian=BHessian, Bdag=Bdag)

    def draw_B(self):
        nr = 20
        nz = 20
        minR = (1 - self.drawZoomOut*self.eps) * self.R0
        maxR = (1 + self.drawZoomOut*self.eps) * self.R0
        minZ = (-self.drawZoomOut*self.k*self.eps) * self.R0
        maxZ = (self.drawZoomOut*self.k*self.eps) * self.R0
        R = np.linspace(minR, maxR, nr)
        Z = np.linspace(minZ, maxZ, nz)
        RR, ZZ = np.meshgrid(R, Z)
        BR = np.zeros([nr, nz])
        Bp = np.zeros([nr, nz])
        Bz = np.zeros([nr, nz])
        for ir in range(nr):
            for iz in range(nz):
                temp = self.compute(np.array([R[ir], 0, Z[iz], 0]))
                BR[ir, iz] = temp.B[0]
                Bp[ir, iz] = temp.B[1]
                Bz[ir, iz] = temp.B[2]

        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True)
        ax = axs[0]
        ax.contourf(RR, ZZ, BR.transpose(), 50, cmap=cm.hot)
        ax.set_title('BR')
        ax.set(xlabel="R [m]", ylabel="Z [m]")

        ax = axs[1]
        ax.contourf(RR, ZZ, Bp.transpose(), 50, cmap=cm.hot)
        ax.set_title('Bphi')
        ax.set(xlabel="R [m]", ylabel="Z [m]")

        ax = axs[2]
        ax.contourf(RR, ZZ, Bz.transpose(), 50, cmap=cm.hot)
        ax.set_title('Bz')
        ax.set(xlabel="R [m]", ylabel="Z [m]")

        fig.show()
