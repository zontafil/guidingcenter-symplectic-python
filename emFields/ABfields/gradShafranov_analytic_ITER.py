# ITER Shafranov equilibrium. See
# https://aip.scitation.org/doi/full/10.1063/1.3328818

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

    def psi(self, x, y):

        psi1 = 1
        psi2 = x**2
        psi3 = y**2 - x**2 * np.log(x)
        psi4 = x**4 - 4*x**2*y**2
        psi5 = 2*y**4 - 9*y**2*x**2 + 3*x**4*np.log(x) - 12*x**2*y**2*np.log(x)
        psi6 = x**6 - 12*x**4*y**2 + 8*x**2*y**4
        psi7 = 8*y**6 - 140*y**4*x**2 + 75*y**2*x**4 -\
            15*x**6*np.log(x) + 180*x**4*y**2*np.log(x) - 120*x**2*y**4*np.log(x)

        ret = x**4 / 8. + A * (0.5 * x**2 * np.log(x) - x**4 / 8.) +\
            c1 * psi1 + c2*psi2 + c3*psi3 + c4*psi4 + c5*psi5 + c6*psi6 + c7*psi7

        return ret

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
