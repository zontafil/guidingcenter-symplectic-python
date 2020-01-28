import numpy as np
from particle import InitializationType


class Config:
    def __init__(self):
        self.h = 20
        self.nsteps = 1000000
        # self.nsteps = 4000
        self.outFile = "out_guidingcenter.txt"

        # INTEGRATOR
        # self.integrator = "RK4"
        self.integrator = "SymplecticExplicit4_GaugeInvariant"
        # self.integrator = "SymplecticExplicit4"
        # self.integrator = "SymplecticExplicit3"
        # self.integrator = "SymplecticExplicit3GaugeFree"
        # self.integrator = "VariationalMidpoint"
        # self.integrator = "SymplecticImplicit1"

        # INITIALIZATION
        self.initializationType = InitializationType.LAGRANGIAN

        # EM FIELD AND DERIVATIVE ALGORITHM
        # self.AB_dB_Algorithm = "finiteDFromA"
        self.AB_dB_Algorithm = "finiteDFromAB"
        # self.AB_dB_Algorithm = "GradShafranovSplineABdB"
        # self.AB_dB_Algorithm = "splineField"

        # self.emField = "Tokamak"
        # self.emField = "GradShafranovSplineA"
        self.emField = "GradShafranovSplineAB"

        # INITIAL CONDITIONS
        self.z0 = np.array([1.000000000000000, 1.00000000000000, 0.000000000000000, 3.9E-6])
        # self.z0 = np.array([0.050000000000000, 0.00000000000000, 0.000000000000000, 3.9E-4])

        self.firstGuessIntegrator = "Euler"
        self.implicit_iterations = 2

        self.system = "GuidingCenter"
        self.mu = 2.5E-6
        self.hx = 1E-5
        self.B0 = 1
        self.R0 = 1
        self.q = 2
        self.z1 = None
        self.p1 = None
        self.p0 = None
        self.initSteps = 200
        self.auxiliaryIntegrator = "RK4"
        self.printTimestepMult = 1000
        self.fileTimestepMult = 1
        self.exitOnError = True
        self.errorThreshold = 0.1
        self.eqdskFile = "data/neqdsk_66832"
