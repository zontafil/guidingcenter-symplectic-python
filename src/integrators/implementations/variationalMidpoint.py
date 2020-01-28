# build a variational integrator starting from the continuous lagrangian and
# discretizing it with the midpoint rule
from integrators.variationalIntegrators.variationalTrapezoidal import VariationalTrapezoidal


class VariationalMidpoint(VariationalTrapezoidal):

    def __init__(self, config):
        super().__init__(config, 0.5)
