import numpy as np
from particle import InitializationType


class Config:
    def __init__(self):
        self.h = 5E-7  # timestep [s]

        self.nsteps = 3000
        self.stepsPerOrbit = 138  # only for chart printing

        # INTEGRATOR
        # self.integrator = "SymplecticExplicit4_GaugeInvariant"
        # self.integrator = "SymplecticExplicit4_ThirdOrder"
        # self.integrator = "SymplecticExplicit4"
        # self.integrator = "SymplecticExplicit3"
        # self.integrator = "SymplecticExplicit3GaugeFree"
        # self.integrator = "VariationalMidpoint"
        # self.integrator = "SymplecticImplicit1"
        # self.integrator = "SymplecticImplicit1Test"
        # self.integrator = "SymplecticExplicitTest"
        # self.integrator = "RK4"
        self.integrator = "PauliBoris"
        # self.integrator = "Degenerate3D"
        # self.integrator = "PauliMidpoint"

        # INITIALIZATION
        # self.initializationType = InitializationType.IMPLICIT3D_HAMILTONIAN
        # self.initializationType = InitializationType.IMPLICIT3D_HAMILTONIAN2
        # self.initializationType = InitializationType.IMPLICIT3D
        # self.initializationType = InitializationType.LAGRANGIAN
        # self.initializationType = InitializationType.HAMILTONIAN
        self.initializationType = InitializationType.PAULI_BORIS
        # self.initializationType = InitializationType.MANUAL
        self.initBackwardIterations = 4
        self.initBackWardOrder = 9
        self.initSteps = 100

        # EM FIELD AND DERIVATIVE ALGORITHM
        #
        # self.AB_dB_Algorithm = "finiteDFromA"  # Require EM field
        # self.AB_dB_Algorithm = "finiteDFromAB"  # Require EM field
        self.AB_dB_Algorithm = "GradShafranovSplineABdB"  # GS equilibrium. Require EQSDK
        # self.AB_dB_Algorithm = "ITER"  # Analytic ITER

        # EM fIELD WITHOUT DERIVATIVES. Required from some ABdB algorithms
        #
        # self.emField = "ITERfield"
        # self.emField = "TokamakQin"
        self.emField = ""

        # EQSDK input files
        #
        self.eqdskFile = "./data/DIIID"

        # INITIAL CONDITIONS
        # physical X, norm u
        self.z0 = self.z1 = self.p0 = self.p1 = None
        # self.z0 = np.array([7, 0.0, 0, 0.30])  # ITER
        self.z0 = np.array([1.8, 0.0, 0, 0.00])  # DIII-D

        self.firstGuessIntegrator = "Euler"
        self.implicit_iterations = 5

        self.kb = 1.3806505E-23
        self.m = 9.108E-31
        self.m = 3.34358E-27
        # self.m = 9.108E-28
        self.q = 1.6021E-19

        self.E0 = 100.  # E0 in keV
        self.pitch = 0.3
        self.computeMuVfromPitch = True

        self.system = "GuidingCenter"
        self.outFile = "./out/out_guidingcenter.txt"
        self.hx = 1E-5
        self.B0 = 5.3
        self.R0 = 6.2
        self.psi0 = 80
        self.auxiliaryIntegrator = "RK4"
        self.printTimestepMult = 1000
        self.fileTimestepMult = np.ceil(self.nsteps / 99999)
        self.exitOnError = True
        self.errorThreshold = 0.5
        self.psi_degree = 4
        self.f_degree = 4
        self.kpsi = 1

        self.debugBfield = False
        self.debugBfieldFile = "./out/Bdebug.txt"

        self.drawBandPsi = False
