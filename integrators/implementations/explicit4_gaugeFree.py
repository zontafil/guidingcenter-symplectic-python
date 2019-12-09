from integrators.integrator import Integrator
from particleUtils import z2p2
import numpy as np


class SymplecticExplicit4_GaugeFree(Integrator):
    def __init__(self, config):
        super().__init__(config)
        self.mu = self.config.mu
        # self.file = open("bhes.txt", "w+")

    def stepForward(self, points, h, t):
        z1 = points.z1
        z0 = points.z0
        ABdB = self.system.fieldBuilder.compute(z1)
        # BHessian = np.zeros([3, 3])
        BHessian = np.array(ABdB.BHessian)

        # DEBUG - REMOVE WHEN EVERYTHING OK
        # if self.config.BHessian_num_4:
        #     for j in range(3):
        #         z1p1 = np.array(z1)
        #         z1m1 = np.array(z1)
        #         z1p2 = np.array(z1)
        #         z1m2 = np.array(z1)
        #         z1m1[j] -= self.config.hx
        #         z1p1[j] += self.config.hx
        #         z1m2[j] -= 2 * self.config.hx
        #         z1p2[j] += 2 * self.config.hx
        #         Bp1 = self.system.fieldBuilder.compute(z1p1)
        #         Bp2 = self.system.fieldBuilder.compute(z1p2)
        #         Bm1 = self.system.fieldBuilder.compute(z1m1)
        #         Bm2 = self.system.fieldBuilder.compute(z1m2)
        #         BHessian[:, j] = (1/12 * Bm2.Bgrad - 2/3 * Bm1.Bgrad +
        #                           2/3 * Bp1.Bgrad - 1/12 * Bp2.Bgrad) / self.config.hx

        # if self.config.BHessian_num:
        #     for j in range(3):
        #         z1p = np.array(z1)
        #         z1m = np.array(z1)
        #         z1m[j] -= self.config.hx
        #         z1p[j] += self.config.hx
        #         B1 = self.system.fieldBuilder.compute(z1p)
        #         B0 = self.system.fieldBuilder.compute(z1m)
        #         BHessian[:, j] = 0.5*(B1.Bgrad - B0.Bgrad) / self.config.hx

        # file = open("bhes.txt", "a")
        # print("====")
        # dBhes = ABdB.BHessian - BHessian
        # theta = np.arctan(points.z1[1]/points.z1[0])
        # self.file.write("{} ".format(t))
        # self.file.write("{:.12f} ".format(np.sqrt(points.z1[0]**2 + points.z1[1]**2)))
        # self.file.write("{:.12f} ".format(theta))
        # for i in range(3):
        #     self.file.write("{:.12f} ".format(points.z1[i]))
        # for i in range(3):
        #     for j in range(3):
        #         self.file.write("{:.12f} ".format(BHessian[i, j]))
        # self.file.write("\n")

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
