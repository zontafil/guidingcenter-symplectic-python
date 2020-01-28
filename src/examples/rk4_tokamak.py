import numpy as np
from particle import InitializationType


class Config:
    def __init__(self):
        self.h = 400
        self.nsteps = 1000000
        self.outFile = "out_guidingcenter.txt"
        self.integrator = "RK4"
        self.system = "GuidingCenter"
        self.mu = 2.5E-6
        self.hx = 1E-5
        self.AB_dB_Algorithm = "finiteDFromAB"
        self.emField = "Tokamak"
        self.B0 = 1
        self.R0 = 1
        self.q = 2
        # self.z0 = np.array([1.000000000000000, 1.00000000000000, 0.000000000000000, 3.9E-4])
        self.z0 = np.array([0.050000000000000, 0.00000000000000, 0.000000000000000, 3.9E-4])
        self.z1 = None
        self.p1 = None
        self.p0 = None
        self.initializationType = InitializationType.MANUAL
        self.initSteps = 100
        self.auxiliaryIntegrator = "RK4"
        self.printTimestepMult = 1000
        self.fileTimestepMult = 1
        self.exitOnError = True
        self.errorThreshold = 0.1
