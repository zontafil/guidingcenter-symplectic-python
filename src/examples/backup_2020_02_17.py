import numpy as np
from particle import InitializationType


class Config:
    def __init__(self):
        self.h = 1
        self.nsteps = 1000
        # self.nsteps = 100000

        # INTEGRATOR
        self.integrator = "RK4"
        # self.integrator = "SymplecticExplicit4_GaugeInvariant"
        # self.integrator = "SymplecticExplicit4_ThirdOrder"
        # self.integrator = "SymplecticExplicit4"
        # self.integrator = "SymplecticExplicit3"
        # self.integrator = "SymplecticExplicit3GaugeFree"
        # self.integrator = "VariationalMidpoint"
        # self.integrator = "SymplecticImplicit1"
        # self.integrator = "SymplecticImplicit1Test"
        # self.integrator = "SymplecticExplicitTest"

        # INITIALIZATION
        self.initializationType = InitializationType.LAGRANGIAN
        self.initBackwardIterations = 6
        self.initBackWardOrder = 9
        self.initSteps = 100

        # EM FIELD AND DERIVATIVE ALGORITHM
        # self.AB_dB_Algorithm = "finiteDFromA"
        # self.AB_dB_Algorithm = "finiteDFromAB"
        # self.AB_dB_Algorithm = "GradShafranovSplineABdB"
        # self.AB_dB_Algorithm = "GradShafranovAnalyticABdB"
        # self.AB_dB_Algorithm = "splineField"
        self.AB_dB_Algorithm = "ITER"

        # self.emField = "Tokamak"
        # self.emField = "GradShafranovSplineA"
        # self.emField = "GradShafranovAnalyticAB"
        self.emField = "GradShafranovSplineAB"
        # self.emField = "ITERfield"

        # INITIAL CONDITIONS
        self.z0 = self.z1 = self.p0 = self.p1 = None
        self.z0 = np.array([1.000000000000000, 1.00000000000000, 0.000000000000000, 3.9E-6])  # shafranov DIII-D
        # self.z0 = np.array([1.000000000000000, 1.00000000000000, 0.000000000000000, 0])  # shafranov DIII-D
        # self.z0 = np.array([0.050000000000000, 0.00000000000000, 0.000000000000000, 3.9E-4])
        # self.z1 = np.array([5.0000142204e-02,  1.0623515018e-03, -1.6850129363e-04,  3.8999661382e-04])

        # TEST
        # self.z0 = np.array([0.000760285680846674, 1.55016358548, -0.27781898273, -0.256143210564])

        # self.z0 = np.array([1.07073211759, -1.05476571403, 0.371281329335, -0.000665821356586])  # debug
        # self.z0 = np.array([1.07073211759, -1.05476571403, 0.371281329335, -0.006624821356586])  # ITER mu5E-3 unstable threshold -5
        self.z0 = np.array([1.07073211759, -1.05476571403, 0.371281329335, -0.0665821356586])  # ITER mu5E-3 unstable
        # self.z0 = np.array([1.07073211759, -1.05476571403, 0.371281329335, 0])  # ITER mu5E-3

        # self.z0 = np.array([1.14735728e+00,   5.42985243e-01,  -9.16118194e-01,  0])  # test iter

        # self.z0 = np.array([0.54488072983, -1.30474450045, 0.178514411629, -1.44103593189e-06])  # debug 0 energy

        self.firstGuessIntegrator = "Euler"
        self.implicit_iterations = 3

        self.system = "GuidingCenter"
        self.outFile = "./out/out_guidingcenter.txt"
        self.mu = 5E-3
        self.hx = 1E-5
        self.B0 = 1
        self.R0 = 1
        self.q = 2
        self.auxiliaryIntegrator = "RK4"
        self.printTimestepMult = 1000
        self.fileTimestepMult = np.ceil(self.nsteps / 99999)
        self.exitOnError = True
        self.errorThreshold = 0.5
        self.eqdskFile = "./data/neqdsk_66832"
        self.psi_degree = 4
        self.f_degree = 4
        self.kpsi = 1
        self.debugBfield = False
        self.stepsPerOrbit = 10000 / self.h
