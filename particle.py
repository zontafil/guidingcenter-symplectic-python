from enum import Enum
from integrators.integratorFactory import integratorFactory
from particleUtils import z0z1p0p1
import numpy as np
from systems.systemFactory import systemFactory
from scipy.interpolate import KroghInterpolator


class InitializationType(Enum):
    MANUAL = 0
    LAGRANGIAN = 1
    HAMILTONIAN = 2
    MANUAL_Z0Z1 = 3


class Particle:
    def __init__(self, config, z0, p0, z1, p1):
        self.config = config
        self.integrator = integratorFactory(config.integrator, config)
        self.h = config.h

        # init particle initial conditions
        self.z1 = z1
        self.z0 = z0
        self.p1 = p1
        self.p0 = p0

        if config.debugBfield:
            self.system = systemFactory(config.system, config)
            self.Bout = open("./Bdebug.txt", "w+")
            self.Bout.write("t Ax Ay Az Bx By Bz\n")

    def r1(self):
        return np.sqrt(self.z1[0]**2 + self.z1[1]**2)

    def computeEnergyError(self):
        self.E1 = self.integrator.system.hamiltonian(self.z1)
        self.dE1 = (self.E1 - self.Einit) / self.Einit

    def stepForward(self, t):
        # compute next time step
        points = self.integrator.stepForward(self.getPoints(), self.h)

        # shift values
        self.z0 = self.z1
        self.p0 = self.p1
        self.dE0 = self.dE1
        self.E0 = self.E1
        self.z1 = points.z2
        self.p1 = points.p2

        # print magnetic field along the particle orbit
        if self.config.debugBfield:
            field = self.system.fieldBuilder.compute(points.z2)
            self.Bout.write("{} {} {} {} {} {} {}\n".format(t,
                            field.A[0], field.A[1], field.A[2], field.B[0], field.B[1], field.B[2]))

        # compute conserved quantitiesEerr0
        self.computeEnergyError()

    def initialize(self, init_type):

        # initialize the particle
        if init_type == InitializationType.LAGRANGIAN:
            # initialize the particle with the help of an auxiliary integrator,
            # in case of the initial conditions are not sufficient for the discrete problem.
            # see PDF, paragraph 6.3

            # compute z1 from z0 using an auxiliary integrator
            if self.config.initSteps <= 0:
                raise Exception("init_steps must be > 0")
            auxiliaryIntegrator = integratorFactory(self.config.auxiliaryIntegrator, self.config)
            self.z1 = np.array(self.z0)
            for i in range(self.config.initSteps):
                points = z0z1p0p1(z1=self.z1, z0=None, p0=None, p1=None)
                points = auxiliaryIntegrator.stepForward(points, self.h / self.config.initSteps)
                self.z1 = points.z2

            # try to compute initial p1 if the integrator supports position-momentum form
            if hasattr(self.integrator.__class__, "legendreRight"):
                self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)

        elif init_type == InitializationType.MANUAL:
            self.z1 = self.z0
            self.p1 = self.p0

        elif init_type == InitializationType.HAMILTONIAN:
            # initialize by imposing the continuous momenta to the discrete space
            self.p0 = self.integrator.system.momentum(self.z0)

            # try to compute z1 using discrete legendre transforms if the integrator supports them
            if (hasattr(self.integrator.__class__, "legendreRight") and
               hasattr(self.integrator.__class__, "legendreLeftInverse")):
                points_z0z1p0p1 = z0z1p0p1(z0=None, p0=None, z1=self.z0, p1=self.p0)
                self.z1 = self.integrator.legendreLeftInverse(points_z0z1p0p1, self.h)
                self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)
            else:
                self.z1 = np.array(self.z0)
                self.p1 = np.array(self.p0)
        elif init_type == InitializationType.MANUAL_Z0Z1:
            if hasattr(self.integrator.__class__, "legendreLeft"):
                self.p0 = self.integrator.legendreLeft(self.z0, self.z1, self.h)
            if hasattr(self.integrator.__class__, "legendreRight"):
                self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)

        # compute initial energy
        self.Einit = self.integrator.system.hamiltonian(self.z0)
        self.computeEnergyError()

    """Initialize decreasing even-odd splitting by interpolating even N steps
    This is one iteration that is suppose to converge after 2-4 iterations
    To be used with a first auxiliary initialization
    """
    def backwardInitializationIteration(self, order):
        even_points = np.zeros([order, 4])
        ts = np.zeros(order)
        for i in range(order):
            ts[i] = i * 2
        even_points[0, :] = np.array(self.z0)
        for i in range(order * 2 - 2):
            self.stepForward(self.h)
            if i % 2 == 0:
                even_points[int(i / 2) + 1, :] = np.array(self.z1)

        for i in range(4):
            interp = KroghInterpolator(ts, even_points[:, i])
            self.z1[i] = interp(1)

        self.z0 = np.array(even_points[0, :])

        if hasattr(self.integrator.__class__, "legendreLeft"):
            self.p0 = self.integrator.legendreLeft(self.z0, self.z1, self.h)
        if hasattr(self.integrator.__class__, "legendreRight"):
            self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)

        self.computeEnergyError()

    def getPoints(self):
        return z0z1p0p1(z0=self.z0, z1=self.z1, p0=self.p0, p1=self.p1)
