# Symplectic integrator for guiding center
# see paragraph 6.4.1
from integrators.implicitIntegrator import ImplicitIntegrator
import numpy as np


class SymplecticImplicit1Test(ImplicitIntegrator):
    def __init__(self, config):
        super().__init__(config)
        self.mu = config.mu

    def f(self, z0, z1, z2, h):

        zm0 = (z0 + z1) / 2.
        zm1 = (z1 + z2) / 2.
        um0 = (z0[3] + z1[3]) / 2.
        um1 = (z1[3] + z2[3]) / 2.

        field_zm0 = self.system.fieldBuilder.compute(zm0)
        field_zm1 = self.system.fieldBuilder.compute(zm1)

        ABdB = self.system.fieldBuilder.compute(z1)

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

        ret = 0.5 * np.dot(omega1, z2 - z0)

        ret[:3] -= h/2.*self.mu*field_zm0.Bgrad
        ret[:3] -= h/2. * self.mu * field_zm1.Bgrad
        # p1[:3] = 0.5*np.dot(np.transpose(field.Adag_jac), dx) + field.Adag - h/2.*self.mu*field.Bgrad
        # p1[3] = 0.5*np.dot(field.b, dx) - h/2. * um
        ret[3] -= h/2. * um0
        ret[3] -= h/2. * um1

        return ret
