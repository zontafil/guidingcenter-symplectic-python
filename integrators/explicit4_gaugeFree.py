from integrators.integrator import Integrator
from particleUtils import z2p2
import numpy as np


class SymplecticExplicit4_GaugeFree(Integrator):
    def __init__(self, config):
        super().__init__(config)
        self.mu = self.config.mu

    def stepForward(self, points, h):
        z1 = points.z1
        z0 = points.z0
        ABdB = self.system.fieldBuilder.compute(z1)
        # BHessian = np.zeros([3, 3])
        BHessian = ABdB.BHessian

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
