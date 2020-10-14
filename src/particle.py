from enum import Enum
from integrators.integratorFactory import integratorFactory
from particleUtils import z0z1p0p1
import numpy as np
from systems.systemFactory import systemFactory
from scipy.interpolate import KroghInterpolator
import matplotlib.pyplot as plt
import sys


class InitializationType(Enum):
    MANUAL = 0
    LAGRANGIAN = 1
    HAMILTONIAN = 2
    MANUAL_Z0Z1 = 3
    PAULI_BORIS = 4
    IMPLICIT3D = 5
    IMPLICIT3D_HAMILTONIAN = 6


class Particle:
    def __init__(self, config, z0, p0, z1, p1):
        self.config = config
        self.integrator = integratorFactory(config.integrator, config)
        self.system = systemFactory(config.system, config)
        self.c = 2.998E8

        if config.drawBandPsi:
            self.system.fieldBuilder.draw_B()
            self.system.fieldBuilder.draw_psirz()
            plt.show()
            sys.exit(0)

        # init particle initial conditions
        self.z1 = z1
        self.z0 = z0
        self.p1 = p1 if p1 is not None else np.zeros(4)
        self.p0 = p0 if p0 is not None else np.zeros(4)

        # compute initial u0 and mu0 from pitch and total initial velocity
        if self.config.computeMuVfromPitch:
            self.config.v0 = np.sqrt(self.config.E0 * 1000. / 6.241509e18 * 2. / self.config.m) / self.c
            field = self.system.fieldBuilder.compute(self.z0)
            self.config.mu = self.config.v0**2 / 2 / field.Bnorm * (1 - self.config.pitch**2)
            self.z0[3] = self.config.pitch * self.config.v0

            print("Vpar computed from pitch and energy: {}".format(self.z0[3]))

            A0 = self.config.m*self.c/self.config.q  # 1.7E-3, A_norm = A / A0
            B_real = field.Bnorm * A0
            larm_w = self.config.q * B_real / self.config.m
            larm_T = 2. * 3.14 / larm_w
            print("larmor period {}".format(larm_T))
            print("B {}".format(B_real))

        # normalize variables
        self.h = self.config.h * self.c
        self.config.h = self.h

        if config.debugBfield:
            self.Bout = open(config.debugBfieldFile, "w+")
            self.Bout.write("t muB 05u2 Ax Ay Az Bx By Bz Bxx Bxy Bxz Byx Byy Byz Bzx Bzy Bzz\n")

    def r1(self):
        return np.sqrt(self.z1[0]**2 + self.z1[1]**2)

    def r0(self):
        return np.sqrt(self.z0[0]**2 + self.z0[1]**2)

    def computeEnergyError(self):
        BdB1 = self.integrator.system.fieldBuilder.compute(self.z1)
        self.E1 = self.integrator.system.hamiltonian(self.z1, BdB1)
        self.dE1 = (self.E1 - self.Einit) / self.Einit
        self.pphi1 = self.integrator.system.toroidalMomentum(self.z1, BdB1)
        self.dpphi1 = (self.pphi1 - self.pphi_init) / self.pphi_init

    def stepForward(self, t):
        # compute next time step
        points = self.integrator.stepForward(self.getPoints(), self.h)

        # shift values
        self.z0 = self.z1
        self.p0 = self.p1
        self.dE0 = self.dE1
        self.E0 = self.E1
        self.dpphi0 = self.dpphi1
        self.pphi0 = self.pphi1
        self.z1 = points.z2
        self.p1 = points.p2 if points.p2 is not None else np.zeros(4)

        # print magnetic field along the particle orbit
        if self.config.debugBfield:
            field = self.system.fieldBuilder.compute(points.z2)
            self.Bout.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(t,
                            self.config.mu * field.Bnorm, 0.5 * self.z1[3]**2,
                            field.A[0], field.A[1], field.A[2], field.B[0], field.B[1], field.B[2],
                            field.BHessian[0][0], field.BHessian[0][1], field.BHessian[0, 2],
                            field.BHessian[1][0], field.BHessian[1][1], field.BHessian[1, 2],
                            field.BHessian[2][0], field.BHessian[2][1], field.BHessian[2, 2]))

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

        elif init_type == InitializationType.PAULI_BORIS:
            field = self.system.fieldBuilder.compute(self.z0)
            self.p0[:3] = field.b * self.z0[3]

            self.z1 = np.array(self.z0)
            self.p1 = np.array(self.p0)

            z2p2 = self.integrator.stepForward(self.getPoints(), self.config.h)
            self.z0 = np.array(self.z1)
            self.p0 = np.array(self.p1)
            self.z1 = z2p2.z2
            self.p1 = z2p2.p2

        elif init_type == InitializationType.IMPLICIT3D:
            # Lagrangian initialization for a 3D guiding center integrator.
            # find z1 from z0 with an auxliary intewgrator.
            # Find p1 from x0, x1 using the Discrete right Legendre transform
            auxiliaryIntegrator = integratorFactory(self.config.auxiliaryIntegrator, self.config)

            z0 = np.array(self.z0)
            for i in range(100):
                points = z0z1p0p1(z1=z0, z0=None, p0=None, p1=None)
                points = auxiliaryIntegrator.stepForward(points, self.h / 100)
                z0 = points.z2
            self.z1 = z0

            self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)
            self.p0 = self.integrator.legendreLeft(self.z0, self.z1, self.h)

            self.integrator.updateVparFromPoints(self.getPoints())

        elif init_type == InitializationType.IMPLICIT3D_HAMILTONIAN:
            # Initialization for the 3D guiding center integrator.
            # Find p0 from x0 using the Dirac constraints p = A + b(b.dot(v)) ...
            # ... and assuming b.dot(v) = u0, with u0 initial condition given as input
            self.z1 = np.array(self.z0)
            field = self.system.fieldBuilder.compute(self.z1)
            self.p1 = np.zeros(4)
            self.p1[:3] = self.z1[3] * field.b + field.A

            z2p2 = self.integrator.stepForward(self.getPoints(), self.config.h)
            self.z0 = np.array(self.z1)
            self.p0 = np.array(self.p1)
            self.z1 = z2p2.z2
            self.p1 = z2p2.p2
            self.integrator.updateVparFromPoints(self.getPoints())

        self.saveInitialEnergy()

    def saveInitialEnergy(self):
        # compute initial energy
        self.Einit = self.integrator.system.hamiltonian(self.z0)
        self.pphi_init = self.integrator.system.toroidalMomentum(self.z0)
        self.pphi0 = self.pphi_init
        self.dpphi0 = 0
        self.E0 = self.Einit
        self.dE0 = 0
        self.computeEnergyError()

    """Initialize decreasing even-odd splitting by interpolating even N steps
    This is one iteration that is suppose to converge after 2-4 iterations
    To be used with a first auxiliary initialization
    """
    def backwardInitializationIteration(self, order):

        # backup initial energy
        E0 = self.E0
        dE0 = self.dE0
        pphi_init = self.pphi_init
        pphi0 = self.pphi0

        even_points = np.zeros([order, 4])
        ts = np.zeros(order)
        for i in range(order):
            ts[i] = i * 2
        even_points[0, :] = np.array(self.z0)
        for i in range(order * 2 - 2):
            self.stepForward(0)
            if i % 2 == 0:
                even_points[int(i / 2) + 1, :] = np.array(self.z1)

        for i in range(4):
            interp = KroghInterpolator(ts, even_points[:, i])
            self.z1[i] = interp(1)
        # self.z1 = (self.z0 + even_points[1., :]) / 2.

        # restore initial position and energy
        self.z0 = np.array(even_points[0, :])
        self.E0 = E0
        self.dE0 = dE0
        self.pphi_init = pphi_init
        self.pphi0 = pphi0

        if hasattr(self.integrator.__class__, "legendreLeft"):
            self.p0 = self.integrator.legendreLeft(self.z0, self.z1, self.h)
        if hasattr(self.integrator.__class__, "legendreRight"):
            self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)
        if hasattr(self.integrator.__class__, "updateVparFromPoints"):
            self.integrator.updateVparFromPoints(self.getPoints())

        self.saveInitialEnergy()
        self.computeEnergyError()

    def getPoints(self):
        return z0z1p0p1(z0=self.z0, z1=self.z1, p0=self.p0, p1=self.p1)
