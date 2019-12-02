# guiding center system, degenerate version
# --> dimension must be 8 or 6
# --> H(q,p) = H(q)
# --> momentum(q0,q1) = momentum(q0)

# Use with variational phase-space symplectic integrators

import numpy as np
from systems.system import System
from emFields.AB_dBfields.AB_dBfactory import AB_dB_FieldFactory


class GuidingCenter(System):
    def __init__(self, config):
        self.mu = config.mu
        self.hx = config.hx  # step for numerical derivative

        # build an em guiding field.
        self.fieldBuilder = AB_dB_FieldFactory(config.AB_dB_Algorithm, config)

    def momentum(self, z):

        # degenerate momenta for guiding center 8D.
        # see paragraph 6.3 of PDF

        p = np.zeros(4)
        ABdB = self.fieldBuilder.compute(z)
        p[:3] = ABdB.A + z[3:] * ABdB.b

        return p

    def hamiltonian(self, z):
        # guiding center hamiltonian
        ABdB = self.fieldBuilder.compute(z)
        return (0.5*z[3]**2 + self.mu * ABdB.Bnorm)

    def lagrangian(self, z, v):
        # guiding center lagrangian
        ABdB = self.fieldBuilder.compute(z)
        return (np.dot(ABdB.Adag, v[:3]) - (0.5*z[3]**2 + self.mu * ABdB.Bnorm))

    def f_eq_motion(self, z):
        ABdB = self.fieldBuilder.compute(z)

        ret = np.zeros(4)
        ret[:3] = (z[3] * ABdB.Bdag + np.cross(ABdB.b, self.mu * ABdB.Bgrad)) / np.dot(ABdB.b, ABdB.Bdag)
        ret[3] = - np.dot(ABdB.Bdag, self.mu * ABdB.Bgrad) / np.dot(ABdB.b, ABdB.Bdag)

        return ret
