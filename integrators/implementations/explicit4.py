# Symplectic integrator for guiding center, explicit 3 (paragraph 6.4.6)
import numpy as np
from integrators.variationalIntegrators.variationalIntegrator import VariationalIntegrator


class SymplecticExplicit4(VariationalIntegrator):
    def __init__(self, config):
        self.mu = config.mu
        super().__init__(config)

    def legendreRight(self, z0, z1, h):

        field = self.system.fieldBuilder.compute(z1)

        M = np.zeros([4, 4])
        p1 = np.zeros(4)

        # BUILD M
        M[0, 1] = field.Bdag[2]
        M[0, 2] = -field.Bdag[1]
        M[1, 0] = -field.Bdag[2]
        M[1, 2] = field.Bdag[0]
        M[2, 0] = field.Bdag[1]
        M[2, 1] = -field.Bdag[0]
        M[0, 3] = -field.b[0]
        M[1, 3] = -field.b[1]
        M[2, 3] = -field.b[2]
        M[3, 0] = field.b[0]
        M[3, 1] = field.b[1]
        M[3, 2] = field.b[2]
        M[0, 0] = M[1, 1] = M[2, 2] = M[3, 3] = 0

        grad2h = np.zeros([4, 4])
        grad2h[:3, :3] = self.mu*field.BHessian
        grad2h[3, 3] = 1.
        M += h/2. * grad2h

        M /= 2.

        dq = z1 - z0
        Q = np.dot(M, dq)

        p1 = np.zeros(4)
        p1[:3] = Q[:3] + field.Adag - h/2.*self.mu*field.Bgrad
        p1[3] = Q[3] - h/2.*z1[3]

        return p1

    def legendreLeftInverse(self, points_z0z1p0p1, h):

        z1 = points_z0z1p0p1.z1
        p1 = points_z0z1p0p1.p1

        field = self.system.fieldBuilder.compute(z1)

        W = np.zeros(4)
        M = np.zeros([4, 4])

        # BUILD M
        M[0, 1] = field.Bdag[2]
        M[0, 2] = -field.Bdag[1]
        M[1, 0] = -field.Bdag[2]
        M[1, 2] = field.Bdag[0]
        M[2, 0] = field.Bdag[1]
        M[2, 1] = -field.Bdag[0]
        M[0, 3] = -field.b[0]
        M[1, 3] = -field.b[1]
        M[2, 3] = -field.b[2]
        M[3, 0] = field.b[0]
        M[3, 1] = field.b[1]
        M[3, 2] = field.b[2]
        M[0, 0] = M[1, 1] = M[2, 2] = M[3, 3] = 0

        grad2h = np.zeros([4, 4])
        grad2h[:3, :3] = self.mu*field.BHessian
        grad2h[3, 3] = 1.
        M -= h/2. * grad2h

        M /= 2.

        # BUILD W
        W[:3] = h/2.*(self.mu*field.Bgrad) + field.Adag - p1[:3]
        W[3] = (h/2.*z1[3] - p1[3])

        Q = np.dot(np.linalg.inv(M), W)

        z2 = np.zeros(4)
        z2[:3] = (z1[:3] + Q[:3])
        z2[3] = (z1[3] + Q[3])

        return z2
