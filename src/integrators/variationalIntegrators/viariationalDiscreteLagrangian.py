# construct a variational integrator starting from a discrete lagrangian.
# the integrator is in general implicit, that's why it's an implementation of VariationalImplicit
from abc import abstractmethod
from integrators.variationalIntegrators.variationalImplicit import VariationalImplicit
import numpy as np


class VariationalDiscreteLagrangian(VariationalImplicit):
    # discrete lagrangian
    @abstractmethod
    def discreteLagrangian(self, z0, z1, h):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.hx = config.hx

    def legendreLeft(self, z0, z1, h):

        p0 = np.zeros(4)
        for i in range(4):
            z0m = np.array(z0)
            z0p = np.array(z0)
            z0m[i] -= self.hx
            z0p[i] += self.hx
            p0[i] = 0.5 * (self.discreteLagrangian(z0m, z1, h) - self.discreteLagrangian(z0p, z1, h)) / self.hx

        return p0

    def legendreRight(self, z0, z1, h):

        p1 = np.zeros(4)
        for i in range(4):
            z1p = np.array(z1)
            z1m = np.array(z1)
            z1m[i] -= self.hx
            z1p[i] += self.hx
            p1[i] = 0.5 * (self.discreteLagrangian(z0, z1p, h) - self.discreteLagrangian(z0, z1m, h)) / self.hx

        return p1
