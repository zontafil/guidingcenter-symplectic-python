from integrators.integrator import Integrator
from particleUtils import z2p2, z0z1p0p1
import numpy as np


class PauliBoris(Integrator):
    def __init__(self, config):
        super().__init__(config)

    def stepForward(self, points, h):
        z1 = points.z1
        p1 = points.p1

        z2 = p1*h + z1
        ABdB = self.system.fieldBuilder.compute(z2)
        B = ABdB.B

        Edagger = - self.config.mu * ABdB.Bgrad

        M = np.zeros([3, 3])
        M[0, 0] = 1. / h
        M[1, 1] = 1. / h
        M[2, 2] = 1. / h
        M[0, 1] = - B[2] / 2.
        M[0, 2] = B[1] / 2
        M[1, 2] = - B[0] / 2.
        M[1, 0] = -M[0, 1]
        M[2, 0] = -M[0, 2]
        M[2, 1] = -M[1, 2]

        w = Edagger + np.cross(p1[:3], B)

        p2 = np.zeros(4)
        p2[:3] = np.dot(np.linalg.inv(M), w) + p1[:3]

        vpar = np.dot(p2[:3], B) / ABdB.Bnorm
        z2[3] = vpar

        return z2p2(z2, p2)

    def legendreLeft(self, z0, z1, h):
        x0 = z0[:3]
        x1 = z1[:3]
        ret = np.zeros(4)

        ret[:3] = (x1 - x0) / h

        return ret

    def legendreRight(self, z0, z1, h):
        x0 = z0[:3]
        x1 = z1[:3]

        p0 = np.zeros(4)
        p0[:3] = (x1 - x0) / h

        points0 = z0z1p0p1(z0=None, p0=None, z1=z0, p1=p0)
        points1 = self.stepForward(points0, h)

        return points1.p2

    def updateVparFromPoints(self, points):
        ABdB = self.system.fieldBuilder.compute(points.z0)
        vpar = np.dot(points.p0[:3], ABdB.B) / ABdB.Bnorm
        points.z0[3] = vpar

        ABdB = self.system.fieldBuilder.compute(points.z1)
        vpar = np.dot(points.p1[:3], ABdB.B) / ABdB.Bnorm
        points.z1[3] = vpar
