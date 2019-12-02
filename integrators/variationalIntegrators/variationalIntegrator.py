# variational integrator:
# the integrator flow is defined by legendre left and right transforms (see paragraph 4.4.2)
# --> (q1,p1) = legendreRight(LegendreLeftInverse(q0,p0))
from integrators.integrator import Integrator
from abc import ABC, abstractmethod
from particleUtils import z2p2


class VariationalIntegrator(Integrator, ABC):

    @abstractmethod
    def legendreRight(self, z0, z1, h):
        pass

    @abstractmethod
    def legendreLeftInverse(self, points_z0z1p0p1, h):
        pass

    # def __init__(self, config):
    #     super().__init__(config)

    def stepForward(self, points, h):
        z1 = points.z1
        z2 = self.legendreLeftInverse(points, h)
        p2 = self.legendreRight(z1, z2, h)

        return z2p2(z2=z2, p2=p2)
