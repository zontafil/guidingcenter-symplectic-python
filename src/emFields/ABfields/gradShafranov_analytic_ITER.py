# ITER Shafranov equilibrium. See
# https://aip.scitation.org/doi/full/10.1063/1.3328818

from emFields.ABfields.emField import EMField
import numpy as np
import collections

SolovevParams = collections.namedtuple("SolovevParams", "A eps delta alpha k")


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class ITERfield(EMField):

    def getITERparams(self):
        return SolovevParams(A=-0.155, eps=0.32, delta=0.33, alpha=np.arcsin(0.33), k=1.7)

    def __init__(self, config):
        self.R0 = config.R0
        self.hx = config.hx

        self.computeCoeff()

    def computeCoeff(self):

        # compute easy params
        p = self.getITERparams()
        self.N1 = - (1. + p.alpha)**2 / (p.eps * p.k**2)
        self.N2 = - (1. - p.alpha)**2 / (p.eps * p.k**2)
        self.N3 = - p.k / (p.eps * np.cos(p.alpha)**2)

        self.Acoeff = p.A
        self.eps = p.eps
        self.delta = p.delta
        self.k = p.k
        self.alpha = p.alpha

        # compute c coeff by Newton iterations
        c = np.zeros(7)
        hx = 1E-5

        for i in range(10):
            f = self.f(c)
            # print("=== Iteration {}".format(i))
            # print("c {}".format(c))
            # print("f {}".format(f))
            Jf = np.zeros([7, 7])
            for j in range(7):
                cp = np.array(c)
                cm = np.array(c)
                cp[j] += hx
                cm[j] -= hx

                df1 = self.f(cp)
                df0 = self.f(cm)

                Jf[:, j] = 0.5*(df1 - df0) / hx

            c = c - np.dot(np.linalg.inv(Jf), f)

        self.c = c

    def f(self, c):
        ret = np.zeros(7)
        ret[0] = self.psiC(1 + self.eps, 0, c)
        ret[1] = self.psiC(1 - self.eps, 0, c)
        ret[2] = self.psiC(1 - self.delta * self.eps, self.k * self.eps, c)
        ret[3] = self.dpsiC_dx(1 - self.delta * self.eps, self.k * self.eps, c)
        ret[4] = self.d2psiC_d2y(1 + self.eps, 0, c) + self.N1 * self.dpsiC_dx(1 + self.eps, 0, c)
        ret[5] = self.d2psiC_d2y(1 - self.eps, 0, c) + self.N2 * self.dpsiC_dx(1 - self.eps, 0, c)
        ret[6] = self.d2psiC_d2x(1 - self.delta * self.eps, self.k * self.eps, c) + self.N3 * self.dpsiC_dy(1 - self.delta * self.eps, self.k * self.eps, c)

        return ret

    def psiC(self, x, y, c):
        psi1 = 1
        psi2 = x**2
        psi3 = y**2 - x**2 * np.log(x)
        psi4 = x**4 - 4*x**2*y**2
        psi5 = 2*y**4 - 9*y**2*x**2 + 3*x**4*np.log(x) - 12*x**2*y**2*np.log(x)
        psi6 = x**6 - 12*x**4*y**2 + 8*x**2*y**4
        psi7 = 8*y**6 - 140*y**4*x**2 + 75*y**2*x**4 -\
            15*x**6*np.log(x) + 180*x**4*y**2*np.log(x) - 120*x**2*y**4*np.log(x)

        ret = x**4 / 8. + self.Acoeff * (0.5 * x**2 * np.log(x) - x**4 / 8.) +\
            c[0] * psi1 + c[1]*psi2 + c[2]*psi3 + c[3]*psi4 +\
            c[4]*psi5 + c[5]*psi6 + c[6]*psi7

        return ret

    def dpsiC_dx(self, x, y, c):
        x1 = x + self.hx
        x0 = x - self.hx
        return 0.5 * (self.psiC(x1, y, c) - self.psiC(x0, y, c)) / self.hx

    def dpsiC_dy(self, x, y, c):
        y1 = y + self.hx
        y0 = y - self.hx
        return 0.5 * (self.psiC(x, y1, c) - self.psiC(x, y0, c)) / self.hx

    def d2psiC_d2y(self, x, y, c):
        y1 = y + self.hx
        y0 = y - self.hx

        return (self.psiC(x, y1, c) - 2.*self.psiC(x, y, c) + self.psiC(x, y0, c)) / self.hx**2

    def d2psiC_d2x(self, x, y, c):
        x1 = x + self.hx
        x0 = x - self.hx

        return (self.psiC(x1, y, c) - 2.*self.psiC(x, y, c) + self.psiC(x0, y, c)) / self.hx**2

    def psi(self, x, y):
        return self.psiC(x, y, self.c)

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
