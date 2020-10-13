# implicit midpoint 3D guiding center integrator
# Based on 4D guiding center Lagrangian, with "u" set to "v.dot(b)"
# see Zonta master thesis, pag. 94
# all derivatives are computed numerically, therefore it is slow
from integrators.integrator import Integrator
from particleUtils import z2p2
import numpy as np
from integrators.explicitIntegratorFactory import explicitIntegratorFactory
from particleUtils import z0z1p0p1


class Degenerate3D(Integrator):
    def __init__(self, config):
        super().__init__(config)
        self.hx = 1E-5
        self.firstGuess = explicitIntegratorFactory(config.firstGuessIntegrator, config)

    def stepForward(self, points, h):

        # project (x,v) into u
        ABdB = self.system.fieldBuilder.compute(points.z1)
        points.z1[3] = np.linalg.norm(points.p1[:3] - ABdB.A)

        x2 = self.LegendreLeftInverse(points.z1, points.p1, h)
        p2 = self.legendreRight(points.z1[:3], x2, h)

        z2 = np.zeros(4)
        z2[:3] = x2

        # project (x,v) into u
        ABdB = self.system.fieldBuilder.compute(z2)
        z2[3] = np.linalg.norm(p2[:3] - ABdB.A)

        return z2p2(z2, p2)

    def LegendreLeftInverse(self, z0, p0, h):

        points = z0z1p0p1(z0=None, z1=z0, p0=None, p1=p0)
        z2p2 = self.firstGuess.stepForward(points, h)
        # print("First guess: {}".format(z2p2.z2))

        x1 = z2p2.z2[:3]
        for i in range(self.config.implicit_iterations):
            # print("==== iteration {}".format(i))
            x1 = self.ImplicitIterationLegendreLeftInverse(z0[:3], p0[:3], x1, h)

        return x1

    def ImplicitIterationLegendreLeftInverse(self, x0, p0, x1, h):

        f = np.zeros(3)
        df1 = np.zeros(3)
        df0 = np.zeros(3)
        Jf = np.zeros([3, 3])
        dx1_0 = np.array(x1)
        dx1_1 = np.array(x1)

        f = p0 - self.legendreLeft(x0, x1, h)[:3]

        for j in range(3):

            dx1_1[j] += self.hx
            dx1_0[j] -= self.hx

            df1 = p0 - self.legendreLeft(x0, dx1_1, h)[:3]
            df0 = p0 - self.legendreLeft(x0, dx1_0, h)[:3]

            Jf[:, j] = 0.5*(df1 - df0) / self.hx

        return x1 - np.dot(np.linalg.inv(Jf), f)

    def legendreLeft(self, z0, z1, h):
        x0 = z0[:3]
        x1 = z1[:3]
        ret = np.zeros(4)

        for i in range(3):
            dx0_1 = np.array(x0)
            dx0_0 = np.array(x0)
            dx0_1[i] += self.hx
            dx0_0[i] -= self.hx

            ret[i] = 0.5 * (self.DiscreteLagrangian(dx0_0, x1, h) - self.DiscreteLagrangian(dx0_1, x1, h)) / self.hx

        return ret

    def DiscreteLagrangian(self, x0, x1, h):
        xalpha = (x0 + x1) / 2.
        dx = (x1 - x0) / h

        return (h * self.Lagrangian(xalpha, dx))

    def Lagrangian(self, x, v):

        z = np.zeros(4)
        z[:3] = x
        ABdB = self.system.fieldBuilder.compute(z)
        u = np.dot(ABdB.b, v)
        return (np.dot(ABdB.A, v) + 0.5*u*u - self.config.mu*ABdB.Bnorm)

    def legendreRight(self, z0, z1, h):
        x0 = z0[:3]
        x1 = z1[:3]
        ret = np.zeros(4)

        for i in range(3):
            dx1_1 = np.array(x1)
            dx1_0 = np.array(x1)
            dx1_1[i] += self.hx
            dx1_0[i] -= self.hx

            ret[i] = 0.5 * (self.DiscreteLagrangian(x0, dx1_1, h) - self.DiscreteLagrangian(x0, dx1_0, h)) / self.hx

        return ret
