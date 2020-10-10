from integrators.integrator import Integrator
from particleUtils import z2p2
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
