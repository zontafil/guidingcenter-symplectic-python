from integrators.integrator import Integrator
from particleUtils import z2p2


class RK4(Integrator):
    def __init__(self, config):
        super().__init__(config)

    def stepForward(self, points, h):
        z1 = points.z1

        k1 = self.system.f_eq_motion(z1)
        k2 = self.system.f_eq_motion(z1 + 0.5*h*k1)
        k3 = self.system.f_eq_motion(z1 + 0.5*h*k2)
        k4 = self.system.f_eq_motion(z1 + h*k3)

        ret = (z1 + 1./6. * h * (k1+2.*k2+2.*k3+k4))
        return z2p2(z2=ret, p2=None)
