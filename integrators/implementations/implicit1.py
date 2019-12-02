# Symplectic integrator for guiding center
# see paragraph 6.4.1
from integrators.variationalIntegrators.variationalImplicit import VariationalImplicit
import numpy as np


class SymplecticImplicit1(VariationalImplicit):
    def __init__(self, config):
        super().__init__(config)
        self.mu = config.mu

    def legendreRight(self, z0, z1, h):

        zm = (z0 + z1) / 2.
        dx = z1[:3] - z0[:3]
        um = (z0[3] + z1[3]) / 2.

        field = self.system.fieldBuilder.compute(zm)

        p1 = np.zeros(4)
        p1[:3] = 0.5*np.dot(np.transpose(field.Adag_jac), dx) + field.Adag - h/2.*self.mu*field.Bgrad
        p1[3] = 0.5*np.dot(field.b, dx) - h/2. * um

        return p1

    def legendreLeft(self, z0, z1, h):

        zm = (z0 + z1) / 2.
        dx = z1[:3] - z0[:3]
        um = (z0[3] + z1[3]) / 2.

        field = self.system.fieldBuilder.compute(zm)

        p0 = np.zeros(4)
        p0[:3] = -0.5*np.dot(np.transpose(field.Adag_jac), dx) + field.Adag + h/2. * self.mu * field.Bgrad
        p0[3] = -0.5*np.dot(field.b, dx) + h/2. * um

        return p0
