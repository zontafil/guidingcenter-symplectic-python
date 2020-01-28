# build a variational integrator starting from the continuous lagrangian and
# discretizing it with the rectangle method (alpha is the position of the height of the rectangle )
from integrators.variationalIntegrators.viariationalDiscreteLagrangian import VariationalDiscreteLagrangian


class VariationalTrapezoidal(VariationalDiscreteLagrangian):

    def __init__(self, config, alpha):
        super().__init__(config)
        self.alpha = alpha

    def discreteLagrangian(self, z0, z1, h):
        qalpha = (1. - self.alpha) * z0 + self.alpha * z1
        dz = (z1 - z0) / h

        return h * self.system.lagrangian(qalpha, dz)
