from enum import Enum
from integrators.integratorFactory import integratorFactory
from particleUtils import z0z1p0p1
import numpy as np
# from emFields.AB_dBfields.finiteDFromAB import FiniteDFromAB
# from emFields.AB_dBfields.gradShafranov_ABdB import GradShafranov_ABdB


class InitializationType(Enum):
    MANUAL = 0
    LAGRANGIAN = 1
    HAMILTONIAN = 2
    MANUAL_Z0Z1 = 3


class Particle:
    def __init__(self, config):
        self.config = config
        self.integrator = integratorFactory(config.integrator, config)
        self.h = config.h

    def r1(self):
        return np.sqrt(self.z1[0]**2 + self.z1[1]**2)

    def computeEnergyError(self):
        self.E1 = self.integrator.system.hamiltonian(self.z1)
        self.dE1 = (self.E1 - self.Einit) / self.Einit

    def stepForward(self, t):
        # compute next time step
        points = self.integrator.stepForward(self.getPoints(), self.h)

        # DfromAB = FiniteDFromAB(self.config)
        # ShafABdB = GradShafranov_ABdB(self.config)
        # # print("point {}".format(self.z1))
        # print(DfromAB.compute(self.z1))
        # shafBdB = ShafABdB.compute(self.z1)
        # print(shafBdB)

        # r = np.sqrt(self.z1[0]**2 + self.z1[1]**2)
        # z = self.z1[2]
        # theta = np.arctan(self.z1[1]/self.z1[0])
        # r1 = r + self.config.hx
        # r0 = r - self.config.hx
        # z1 = z + self.config.hx
        # z0 = z - self.config.hx
        # theta1 = theta + self.config.hx
        # theta0 = theta - self.config.hx

        # zr1 = np.array([r1*np.cos(theta), r1*np.sin(theta), self.z1[2], self.z1[3]])
        # zr0 = np.array([r0*np.cos(theta), r0*np.sin(theta), self.z1[2], self.z1[3]])
        # zz1 = np.array([r*np.cos(theta), r*np.sin(theta), z1, self.z1[3]])
        # zz0 = np.array([r*np.cos(theta), r*np.sin(theta), z0, self.z1[3]])
        # zp1 = np.array([r*np.cos(theta1), r*np.sin(theta1), z, self.z1[3]])
        # zp0 = np.array([r*np.cos(theta0), r*np.sin(theta0), z, self.z1[3]])
        # shafr1 = ShafABdB.compute(zr1)
        # shafr0 = ShafABdB.compute(zr0)
        # shafz1 = ShafABdB.compute(zz1)
        # shafz0 = ShafABdB.compute(zz0)
        # shafp1 = ShafABdB.compute(zp1)
        # shafp0 = ShafABdB.compute(zp0)
        # # TODO: add gradient_cyl_modB to collections
        # # d2modB_d2R = 0.5 * (shafr1.gradB_cyl[0] - shafr0.gradB_cyl[0]) / self.config.hx - shafBdB.d2modB_d2R
        # # d2modB_dRdz = 0.5 * (shafz1.gradB_cyl[0] - shafz0.gradB_cyl[0]) / self.config.hx - shafBdB.d2modB_dRdz
        # # d2modB_d2z = 0.5 * (shafz1.gradB_cyl[2] - shafz0.gradB_cyl[2]) / self.config.hx - shafBdB.d2modB_d2z
        # d2modB_dxdr = 0.5 * (shafr1.Bgrad[0] - shafr0.Bgrad[0]) / self.config.hx - shafBdB.gradCyl_dmodB_dx[0]
        # d2modB_dxdp = 0.5 * (shafp1.Bgrad[0] - shafp0.Bgrad[0]) / r / self.config.hx - shafBdB.gradCyl_dmodB_dx[1]
        # d2modB_dxdz = 0.5 * (shafz1.Bgrad[0] - shafz0.Bgrad[0]) / self.config.hx - shafBdB.gradCyl_dmodB_dx[2]
        # d2modB_dydr = 0.5 * (shafr1.Bgrad[1] - shafr0.Bgrad[1]) / self.config.hx - shafBdB.gradCyl_dmodB_dy[0]
        # d2modB_dydp = 0.5 * (shafp1.Bgrad[1] - shafp0.Bgrad[1]) / r / self.config.hx - shafBdB.gradCyl_dmodB_dy[1]
        # d2modB_dydz = 0.5 * (shafz1.Bgrad[1] - shafz0.Bgrad[1]) / self.config.hx - shafBdB.gradCyl_dmodB_dy[2]
        # d2modB_dzdr = 0.5 * (shafr1.Bgrad[2] - shafr0.Bgrad[2]) / self.config.hx - shafBdB.gradCyl_dmodB_dz[0]
        # d2modB_dzdp = 0.5 * (shafp1.Bgrad[2] - shafp0.Bgrad[2]) / r / self.config.hx - shafBdB.gradCyl_dmodB_dz[1]
        # d2modB_dzdz = 0.5 * (shafz1.Bgrad[2] - shafz0.Bgrad[2]) / self.config.hx - shafBdB.gradCyl_dmodB_dz[2]

        # shift values
        self.z0 = self.z1
        self.p0 = self.p1
        self.dE0 = self.dE1
        self.E0 = self.E1
        self.z1 = points.z2
        self.p1 = points.p2

        field = self.system.fieldBuilder.compute(points.z2)
        self.Bout.write("{} {} {} {} {} {} {}\n".format(t, field.A[0], field.A[1], field.A[2], field.B[0], field.B[1], field.B[2]))

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
                points_z0z1p0p1 = z0z1p0p1(z0=None, p0=None, z1=self.z0, p1=self.p0)
                self.z1 = self.integrator.legendreLeftInverse(points_z0z1p0p1, self.h)
                self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)
            else:
                self.z1 = np.array(self.z0)
                self.p1 = np.array(self.p0)
        elif init == InitializationType.MANUAL_Z0Z1:
            if hasattr(self.integrator.__class__, "legendreLeft"):
                self.p0 = self.integrator.legendreLeft(self.z0, self.z1, self.h)
            if hasattr(self.integrator.__class__, "legendreRight"):
                self.p1 = self.integrator.legendreRight(self.z0, self.z1, self.h)

        # compute initial energy
        self.Einit = self.integrator.system.hamiltonian(self.z0)
        self.computeEnergyError()

    def getPoints(self):
        return z0z1p0p1(z0=self.z0, z1=self.z1, p0=self.p0, p1=self.p1)
