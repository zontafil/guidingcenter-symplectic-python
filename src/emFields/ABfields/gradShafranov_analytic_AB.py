from emFields.ABfields.emField import EMField
import numpy as np


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class GradShafranov_analytic_AB(EMField):
    def __init__(self, config):
        self.R0 = config.R0
        self.hx = config.hx

    def psi(self, r, z):
        return (r - self.R0)**2 + z**2

    def A(self, x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        z = x[2]

        psi = self.psi(r, z)

        Acyl = np.array([0, psi/r, np.log(r / self.R0)])
        return cyl2cart(Acyl, x)

    def B(self, x):
        R = np.sqrt(x[0]**2 + x[1]**2)
        Z = x[2]
        dpsi_dR = 2 * (R - self.R0)
        dpsi_dz = 2 * Z

        curlA_cyl = np.zeros(3)
        curlA_cyl[0] = - dpsi_dz / R
        curlA_cyl[1] = - 1 / R
        curlA_cyl[2] = dpsi_dR / R

        return cyl2cart(curlA_cyl, x)
