# variational integrator, implicit version.
# compute legendre left inverse using a first guess integrator and newton iterations
from integrators.integratorFactory import explicitIntegratorFactory
from abc import ABC, abstractmethod
from integrators.variationalIntegrators.variationalIntegrator import VariationalIntegrator
import numpy as np


class VariationalImplicit(ABC, VariationalIntegrator):

    @abstractmethod
    def legendreLeft(self, z0, z1, h):
        pass

    def __init__(self, config):
        self.firstGuess = explicitIntegratorFactory(config.firstGuessIntegrator, config)
        self.implicitIterations = config.implicit_iterations
        self.hx = config.hx

    def legendreLeftInverse(self, points_z0z1p0p1, h):

        # compute z1 using a first guess (i.e. RK4)
        points_z2p2 = self.firstGuess.stepForward(points_z0z1p0p1, h)
        z1 = points_z0z1p0p1.z1
        p1 = points_z0z1p0p1.p1
        z2 = points_z2p2.z2

        for i in range(self.implicitIterations):
            # compute z2 from z1, p1 and a first guess of z2
            z2 = self.implicitIterationLegendreLeftInverse(z1, p1, z2, h)

        return z2

    def implicitIterationLegendreLeftInverse(self, z1, p1, z2, h):
        # newton iteration for the legendre left inverse.
        # z2_new = z2 - f/f'

        f = p1 - self.legendreLeft(z1, z2, h)
        Jf = np.zeros([4, 4])

        for j in range(4):

            z2p = np.array(z2)
            z2m = np.array(z2)
            z2p[j] += self.hx
            z2m[j] -= self.hx

            df1 = p1 - self.legendreLeft(z1, z2p, h)
            df0 = p1 - self.legendreLeft(z1, z2m, h)

            Jf[:, j] = 0.5*(df1 - df0) / self.hx

        return z2 - np.dot(np.linalg.inv(Jf), f)
