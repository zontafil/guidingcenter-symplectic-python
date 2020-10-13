# implicit midpoint 3D guiding center integrator
# Based on 4D guiding center Lagrangian, with "u" set to "v.dot(b)"
# see Zonta master thesis, pag. 94
# all derivatives are computed numerically, therefore it is slow
from integrators.guidingcenter3D import GuidingcCenter3DIntegrator
import numpy as np


class Degenerate3D(GuidingcCenter3DIntegrator):
    def __init__(self, config):
        super().__init__(config)

    def Lagrangian(self, x, v):

        z = np.zeros(4)
        z[:3] = x
        ABdB = self.system.fieldBuilder.compute(z)
        u = np.dot(ABdB.b, v)
        return (np.dot(ABdB.A, v) + 0.5*u*u - self.config.mu*ABdB.Bnorm)
