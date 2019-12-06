from integrators.integrator import Integrator
from particleUtils import z2p2
import numpy as np


class SymplecticExplicit4_GaugeFree(Integrator):
    def __init__(self, config):
        super().__init__(config)
        self.mu = self.config.mu

    def Bgradnum(self, z):
        Bgrad = np.zeros(3)
        for j in range(3):
            z1p = np.array(z)
            z1m = np.array(z)
            z1p2 = np.array(z)
            z1m2 = np.array(z)
            z1m[j] -= self.config.hx
            z1p[j] += self.config.hx
            z1m2[j] -= 2 * self.config.hx
            z1p2[j] += 2 * self.config.hx
            B1 = self.system.fieldBuilder.compute(z1p)
            B0 = self.system.fieldBuilder.compute(z1m)
            Bp2 = self.system.fieldBuilder.compute(z1p2)
            Bm2 = self.system.fieldBuilder.compute(z1m2)
            # Bgrad[j] = 0.5*(np.linalg.norm(B1.Bnorm) - np.linalg.norm(B0.Bnorm)) / self.config.hx
            Bgrad[j] = (1/12 * Bm2.Bnorm - 2/3 * B0.Bnorm + 2/3 * B1.Bnorm - 1/12 * Bp2.Bnorm) / self.config.hx
        return Bgrad

    def stepForward(self, points, h):
        z1 = points.z1
        z0 = points.z0
        ABdB = self.system.fieldBuilder.compute(z1)
        # BHessian = np.zeros([3, 3])
        BHessian = ABdB.BHessian

        # Bgrad = self.Bgradnum(z1)

        if self.config.BHessian_num_4:
            for j in range(3):
                z1p1 = np.array(z1)
                z1m1 = np.array(z1)
                z1p2 = np.array(z1)
                z1m2 = np.array(z1)
                z1m1[j] -= self.config.hx
                z1p1[j] += self.config.hx
                z1m2[j] -= 2 * self.config.hx
                z1p2[j] += 2 * self.config.hx
                Bp1 = self.system.fieldBuilder.compute(z1p1)
                Bp2 = self.system.fieldBuilder.compute(z1p2)
                Bm1 = self.system.fieldBuilder.compute(z1m1)
                Bm2 = self.system.fieldBuilder.compute(z1m2)
                BHessian[:, j] = (1/12 * Bm2.Bgrad - 2/3 * Bm1.Bgrad + 2/3 * Bp1.Bgrad - 1/12 * Bp2.Bgrad) / self.config.hx

        if self.config.BHessian_num:
            for j in range(3):
                z1p = np.array(z1)
                z1m = np.array(z1)
                z1m[j] -= self.config.hx
                z1p[j] += self.config.hx
                B1 = self.system.fieldBuilder.compute(z1p)
                B0 = self.system.fieldBuilder.compute(z1m)
                BHessian[:, j] = 0.5*(B1.Bgrad - B0.Bgrad) / self.config.hx

        # print("====")
        # print(ABdB.BHessian[0,0])
        # print(BHes[0,0])
        # print(BHes_num_fromB[0,0])
        # print("====")
        # print(ABdB.Bgrad[0])
        # print(Bgrad[0])

        # build omega1
        omega1 = np.zeros([4, 4])
        omega1[0, 1] = - ABdB.Bdag[2]
        omega1[0, 2] = ABdB.Bdag[1]
        omega1[1, 2] = - ABdB.Bdag[0]
        omega1[1, 0] = ABdB.Bdag[2]
        omega1[2, 0] = - ABdB.Bdag[1]
        omega1[2, 1] = ABdB.Bdag[0]
        omega1[0, 3] = ABdB.b[0]
        omega1[1, 3] = ABdB.b[1]
        omega1[2, 3] = ABdB.b[2]
        omega1[3, 0] = - ABdB.b[0]
        omega1[3, 1] = - ABdB.b[1]
        omega1[3, 2] = - ABdB.b[2]

        Hd1 = np.zeros(4)
        Hd1[:3] = self.mu * ABdB.Bgrad
        Hd1[3] = z1[3]

        Hd2 = np.zeros([4, 4])
        Hd2[:3, :3] = self.mu*BHessian
        Hd2[3, 3] = 1.

        M = omega1 / 2. + h / 4. * Hd2
        Q = - h * Hd1 + h / 4. * np.dot(Hd2, 2. * z1 - 2 * z0)

        z2 = np.dot(np.linalg.inv(M), Q) + z0

        return z2p2(z2=z2, p2=None)
