# variational integrator, implicit version.
# compute legendre left inverse using a first guess integrator and newton iterations
from integrators.explicitIntegratorFactory import explicitIntegratorFactory
from abc import abstractmethod
from integrators.integrator import Integrator
from particleUtils import z2p2
import numpy as np


class ImplicitIntegrator(Integrator):

    @abstractmethod
    def f(self, z0, z1, z2, h):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.firstGuess = explicitIntegratorFactory(config.firstGuessIntegrator, config)
        self.implicitIterations = config.implicit_iterations
        self.hx = config.hx

    def stepForward(self, points, h):
        z1 = points.z1
        z0 = points.z0

        # compute z1 using a first guess (i.e. RK4)
        points_z2p2 = self.firstGuess.stepForward(points, h)
        z2 = points_z2p2.z2

        for i in range(self.implicitIterations):
            z2 = self.implicitIteration(z0, z1, z2, h)

        return z2p2(z2=z2, p2=None)

    def implicitIteration(self, z0, z1, z2, h):
        # newton iteration for the legendre left inverse.
        # z2_new = z2 - f/f'

        f = self.f(z0, z1, z2, h)
        Jf = np.zeros([4, 4])

        for j in range(4):

            z2p = np.array(z2)
            z2m = np.array(z2)
            z2p[j] += self.hx
            z2m[j] -= self.hx

            df1 = self.f(z0, z1, z2p, h)
            df0 = self.f(z0, z1, z2m, h)

            Jf[:, j] = 0.5*(df1 - df0) / self.hx

        return z2 - np.dot(np.linalg.inv(Jf), f)
