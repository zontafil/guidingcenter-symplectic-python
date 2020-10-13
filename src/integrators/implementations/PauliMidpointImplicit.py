# Implicit Midpoint integrator for the Pauli particle
# see Qin 2020 - 2006.03818
# all derivatives are computed numerically, therefore it is slow
from integrators.guidingcenter3D import GuidingcCenter3DIntegrator
import numpy as np


class PauliMidpoint(GuidingcCenter3DIntegrator):
    def __init__(self, config):
        super().__init__(config)

    def Lagrangian(self, x, v):

        z = np.zeros(4)
        z[:3] = x
        ABdB = self.system.fieldBuilder.compute(z)
        v2 = np.dot(v, v)
        return (np.dot(ABdB.A, v) + 0.5*v2 - self.config.mu*ABdB.Bnorm)
