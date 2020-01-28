from integrators.implicitIntegrator import ImplicitIntegrator
import numpy as np


class SymplecticExplicit4_ThirdOrder(ImplicitIntegrator):
    def __init__(self, config):
        super().__init__(config)
        self.mu = self.config.mu

    def f(self, z0, z1, z2, h):

        dz1 = z2 - z1
        dz0 = z1 - z0

        ABdB = self.system.fieldBuilder.compute(z1)
        BHessian = np.array(ABdB.BHessian)
        d3B = self.system.fieldBuilder.d3B(z1)

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

        Hd3 = np.zeros([4, 4, 4])
        Hd3[:3, :3, :3] = self.mu * d3B

        ret = 0.5 * np.dot(omega1, z2 - z0)
        ret += h * Hd1
        ret += h / 4 * np.dot(Hd2, z2 - 2 * z1 + z0)

        # third order hamiltonian terms
        ret += h / 16 * np.dot(dz1, np.dot(dz1, Hd3))
        ret += h / 16 * np.dot(dz0, np.dot(dz0, Hd3))

        return ret
