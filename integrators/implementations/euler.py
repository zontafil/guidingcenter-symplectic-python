from integrators.integrator import Integrator
from particleUtils import z2p2


class Euler(Integrator):
    def __init__(self, config):
        super().__init__(config)

    def stepForward(self, points, h):
        z1 = points.z1

        k1 = self.system.f_eq_motion(z1)

        ret = z1 + h * k1
        return z2p2(z2=ret, p2=None)
