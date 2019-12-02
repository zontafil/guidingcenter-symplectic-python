from enum import Enum
from integrators.integratorFactory import integratorFactory
from particleUtils import z0z1p0p1
import numpy as np


class InitializationType(Enum):
    MANUAL = 0
    LAGRANGIAN = 1
    HAMILTONIAN = 2


class Particle:
    def __init__(self, config):
        self.config = config
        self.integrator = integratorFactory(config.integrator, config)
        self.h = config.h

    def r1(self):
        return self.z1[0]**2 + self.z1[1]**2

    def computeEnergyError(self):
        self.E1 = self.integrator.system.hamiltonian(self.z1)
        self.dE1 = (self.E1 - self.Einit) / self.Einit

    def stepForward(self):
        # compute next time step
        points = self.integrator.stepForward(self.getPoints(), self.h)

        # shift values
        self.z0 = self.z1
        self.p0 = self.p1
        self.dE0 = self.dE1
        self.E0 = self.E1
        self.z1 = points.z2
        self.p1 = points.p2

        # compute conserved quantitiesEerr0
        self.computeEnergyError()

    def initialize(self):

        self.z1 = self.config.z1
        self.z0 = self.config.z0
        self.p1 = self.config.p1
        self.p0 = self.config.p0

        # initialize the particle
        init = self.config.initializationType
        if init == InitializationType.LAGRANGIAN:
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

        elif init == InitializationType.MANUAL:
            self.z1 = self.z0
            self.p1 = self.p0

        elif init == InitializationType.HAMILTONIAN:
            # initialize by imposing the continuous momenta to the discrete space
            self.p0 = self.integrator.system.momentum(self.z0)

            # try to compute z1 using discrete legendre transforms if the integrator supports them
            if (hasattr(self.integrator.__class__, "legendreRight") and
               hasattr(self.integrator.__class__, "legendreLeftInverse")):
                self.z1 = self.integrator.legendreLeftInverse(self.z0, self.p0, self.h)
                self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)
            else:
                self.z1 = np.array(self.z0)
                self.p1 = np.array(self.p0)

        # compute initial energy
        self.Einit = self.integrator.system.hamiltonian(self.z0)
        self.computeEnergyError()

    def getPoints(self):
        return z0z1p0p1(z0=self.z0, z1=self.z1, p0=self.p0, p1=self.p1)
