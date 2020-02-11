# ITER Shafranov equilibrium. See
# https://aip.scitation.org/doi/full/10.1063/1.3328818

import numpy as np
import collections
from emFields.AB_dBfields.AB_dBfield import AB_dB_FieldBuilder, ABdBGuidingCenter


SolovevParams = collections.namedtuple("SolovevParams", "A eps delta alpha k")


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class ITERfield(AB_dB_FieldBuilder):

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

    def dpsiy(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = 2*y
        dpsi4 = - 8*x**2*y
        dpsi5 = 8*y**3 - 18*y*x**2 - 24*x**2*y*np.log(x)
        dpsi6 = -24*x**4*y + 32*x**2*y**3
        dpsi7 = 48*y**5 - 560*y**3*x**2 + 150*y*x**4 + 360*x**4*y*np.log(x) - 480*x**2*y**3*np.log(x)

        ret = c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsix(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 2*x
        dpsi3 = - 2*x * np.log(x) - x
        dpsi4 = 4*x**3 - 8*x*y**2
        dpsi5 = - 30*y**2*x + 12*x**3*np.log(x) + 3*x**3 - 24*x*y**2*np.log(x)
        dpsi6 = 6*x**5 - 48*x**3*y**2 + 16*x*y**4
        dpsi7 = -400*y**4*x + 480*y**2*x**3 -15*x**5 -90*x**5*np.log(x) +720*x**3*y**2*np.log(x) -240*x*y**4*np.log(x)

        ret = x**3 / 2. + self.Acoeff * (0.5 * x + x * np.log(x) - x**3 / 2.) +\
            c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsixx(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 2
        dpsi3 = - 2 * np.log(x) -3
        dpsi4 = 12*x**2 - 8*y**2
        dpsi5 = -54*y**2 +21*x**2 + 36*x**2*np.log(x) -24*y**2*np.log(x)
        dpsi6 = 30*x**4 - 144*x**2*y**2 + 16*y**4
        dpsi7 = -640*y**4 +2160*y**2*x**2 -165*x**4 -450*x**4*np.log(x) +2160*x**2*y**2*np.log(x) -240*y**4*np.log(x)

        ret = 3*x**2 / 2. + self.Acoeff * (1.5 + np.log(x) - 3*x**2 / 2.) +\
            c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsiyy(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = 2
        dpsi4 = - 8*x**2
        dpsi5 = 24*y**2 - 18*x**2 - 24*x**2*np.log(x)
        dpsi6 = - 24*x**4 + 96*x**2*y**2
        dpsi7 = 240*y**4 -1680*y**2*x**2 +150*x**4 +360*x**4*np.log(x) -1440*x**2*y**2*np.log(x)

        ret = c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsixy(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = 0
        dpsi4 = - 16*x*y
        dpsi5 = - 60*y*x - 48*x*y*np.log(x)
        dpsi6 = - 96*x**3*y + 64*x*y**3
        dpsi7 = -1600*y**3*x +960*y*x**3 +1440*x**3*y*np.log(x) -960*x*y**3*np.log(x)

        ret = c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsixxy(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = 0
        dpsi4 = - 16*y
        dpsi5 = - 108*y -48*y*np.log(x)
        dpsi6 = -288*x**2*y + 64*y**3
        dpsi7 = - 2560*y**3 + 4320*y*x**2 +4320*x**2*y*np.log(x) -960*y**3*np.log(x)

        ret = c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7


        return ret

    def dpsixyy(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = 0
        dpsi4 = - 16*x
        dpsi5 = - 60*x - 48*x*np.log(x)
        dpsi6 = - 96*x**3 + 192*x*y**2
        dpsi7 = -4800*y**2*x + 960*x**3 +1440*x**3*np.log(x) -2880*x*y**2*np.log(x)

        ret = c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsixxx(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = - 2 / x
        dpsi4 = 24*x
        dpsi5 = +78*x +72*x*np.log(x) -24*y**2/x
        dpsi6 = 120*x**3 - 288*x*y**2
        dpsi7 = 6480*y**2*x -1110*x**3 -1800*x**3*np.log(x) +4320*x*y**2*np.log(x) -240*y**4/x

        ret = 3*x + self.Acoeff * (1./x - 3*x) +\
            c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

        return ret

    def dpsiyyy(self, x, y, c):
        dpsi1 = 0
        dpsi2 = 0
        dpsi3 = 0
        dpsi4 = 0
        dpsi5 = 48*y
        dpsi6 = + 192*x**2*y
        dpsi7 = 960*y**3 -3360*y*x**2 - 2880*x**2*y*np.log(x)

        ret = c[0] * dpsi1 + c[1]*dpsi2 + c[2]*dpsi3 + c[3]*dpsi4 +\
            c[4]*dpsi5 + c[5]*dpsi6 + c[6]*dpsi7

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
        dpsi_dR = self.dpsix(R, Z, self.c)
        dpsi_dz = self.dpsiy(R, Z, self.c)

        curlA_cyl = np.zeros(3)
        curlA_cyl[0] = - dpsi_dz / R
        curlA_cyl[1] = - 1 / R
        curlA_cyl[2] = dpsi_dR / R

        return cyl2cart(curlA_cyl, x)

    def BHessian(self, z):
        ret = np.zeros([3, 3])

        for i in range(3):
            # print("i " + str(i))
            z1 = np.array(z)
            z0 = np.array(z)
            z1[i] += self.hx
            z0[i] -= self.hx
            BdB1 = self.compute(z1)
            BdB0 = self.compute(z0)
            for j in range(3):
                ret[i, j] = 0.5 * (BdB1.Bgrad[j] - BdB0.Bgrad[j]) / self.hx
        
        return ret

    def B_dB_cyl(self, R, Z):

        # interpolate psi and derivatives
        dpsi_dR = self.dpsix(R, Z, self.c)
        dpsi_dz = self.dpsiy(R, Z, self.c)
        d2psi_dR2 = self.dpsixx(R, Z, self.c)
        d2psi_dRdz = self.dpsixy(R, Z, self.c)
        d2psi_dz2 = self.dpsiyy(R, Z, self.c)
        d3psi_d2Rdz = self.dpsixxy(R, Z, self.c)
        d3psi_dRd2z = self.dpsixyy(R, Z, self.c)
        d3psi_d3z = self.dpsiyyy(R, Z, self.c)
        d3psi_d3R = self.dpsixxx(R, Z, self.c)

        # print("analytic")
        # print(d3psi_d3z)
        # print("numerical")
        # numer = 0.5*(self.dpsiyy(R,Z+self.hx,self.c) - self.dpsiyy(R,Z-self.hx,self.c))/self.hx
        # print(numer)


        # evaluate the magnetic field
        BR = -dpsi_dz/R
        Bp = -1 / R
        Bz = dpsi_dR/R
        # evaluate the derivatives
        dBR_dR = dpsi_dz/(R**2)-d2psi_dRdz/R
        dBR_dp = 0.
        dBR_dz = -d2psi_dz2/R
        dBp_dR = 1/(R**2)
        dBp_dp = 0.
        dBp_dz = 0.
        dBz_dR = -dpsi_dR/(R**2) + d2psi_dR2/R
        dBz_dp = 0.
        dBz_dz = d2psi_dRdz/R

        d2BR_d2R = -2 * dpsi_dz / (R**3) + 2 * d2psi_dRdz / (R**2) - d3psi_d2Rdz / R
        d2BR_dRdz = d2psi_dz2 / (R**2) - d3psi_dRd2z / R
        d2BR_d2z = -d3psi_d3z / R
        d2Bp_d2R = -2 / (R**3)
        d2Bp_dRdz = 0.
        d2Bp_d2z = 0.
        d2Bz_d2R = 2 * dpsi_dR / (R**3) - 2 * d2psi_dR2 / (R**2) + d3psi_d3R / R
        d2Bz_dRdz = -d2psi_dRdz / (R**2) + d3psi_d2Rdz / R
        d2Bz_d2z = d3psi_dRd2z / R

        return np.array([BR, Bp, Bz,
                         dBR_dR, dBR_dp, dBR_dz,
                         dBp_dR, dBp_dp, dBp_dz,
                         dBz_dR, dBz_dp, dBz_dz,
                         d2BR_d2R, d2BR_dRdz, d2BR_d2z,
                         d2Bp_d2R, d2Bp_dRdz, d2Bp_d2z,
                         d2Bz_d2R, d2Bz_dRdz, d2Bz_d2z])

    def d3B(self, z):
        ret = np.zeros([3, 3, 3])

        for i in range(3):
            # print("i " + str(i))
            z1 = np.array(z)
            z0 = np.array(z)
            z1[i] += self.hx
            z0[i] -= self.hx
            BdB1 = self.compute(z1)
            BdB0 = self.compute(z0)
            for j in range(3):
                for k in range(3):
                    ret[i, j, k] = 0.5 * (BdB1.BHessian[j, k] - BdB0.BHessian[j, k]) / self.hx

        # print("\n\n===== D3B")
        # print(ret)
        # # print(ret[1,0,0])
        # # print(ret[0,1,0])
        # print("=====\n\n")
        return ret

    def compute(self, z):
        x = z[:3]
        r = np.sqrt(x[0]**2 + x[1]**2)
        u = z[3]
        sintheta = x[1] / r
        costheta = x[0] / r

        BdB = self.B_dB_cyl(r, x[2])
        Bcyl = np.array([BdB[0], BdB[1], BdB[2]])

        # build B, B and |B| (cartesian)
        B = cyl2cart(Bcyl, x)
        Bnorm = np.linalg.norm(B)
        b = B / Bnorm

        A = self.A(x)
        Adag = A + u * b

        # build curl B (cyl and cartesian)
        Bcurl_cyl = np.zeros(3)
        Bcurl_cyl[0] = - BdB[8]
        Bcurl_cyl[1] = BdB[5] - BdB[9]
        Bcurl_cyl[2] = Bcyl[1] / r + BdB[6]
        Bcurl = cyl2cart(Bcurl_cyl, x)

        # build grad|B| (cyl and cartesian)
        dB_dR = np.array([BdB[3], BdB[6], BdB[9]])
        dB_dz = np.array([BdB[5], BdB[8], BdB[11]])
        gradB_cyl = np.zeros(3)
        gradB_cyl[0] = np.dot(Bcyl, dB_dR)
        gradB_cyl[1] = 0
        gradB_cyl[2] = np.dot(Bcyl, dB_dz)
        gradB_cyl /= Bnorm
        B_grad = cyl2cart(gradB_cyl, x)

        # build grad(1/|B|) and Bdag
        grad1_B = - B_grad / Bnorm**2
        Bdag = B + u * Bcurl / Bnorm + u * np.cross(grad1_B, B)

        # compute Bjac
        dBx_dr = BdB[3] * costheta - BdB[6] * sintheta
        dBx_dp = - BdB[0] * sintheta / r - BdB[1] * costheta / r
        dBx_dz = BdB[5] * costheta - BdB[8] * sintheta / r
        dBy_dr = BdB[3] * sintheta + BdB[6] * costheta
        dBy_dp = + BdB[0] * costheta / r - BdB[1] * sintheta / r
        dBy_dz = BdB[5] * sintheta + BdB[8] * costheta / r
        dBz_dr = BdB[9]
        dBz_dp = 0
        dBz_dz = BdB[11]
        Bjac = np.zeros([3, 3])
        Bjac[0, :] = cyl2cart(np.array([dBx_dr, dBx_dp, dBx_dz]), x)
        Bjac[1, :] = cyl2cart(np.array([dBy_dr, dBy_dp, dBy_dz]), x)
        Bjac[2, :] = cyl2cart(np.array([dBz_dr, dBz_dp, dBz_dz]), x)

        # compute |B| hessian
        d2B_d2R = np.array([BdB[12], BdB[15], BdB[18]])
        d2B_dRdz = np.array([BdB[13], BdB[16], BdB[19]])
        d2B_d2z = np.array([BdB[14], BdB[17], BdB[20]])
        d2modB_d2R = - np.dot(Bcyl, dB_dR)**2 / (Bnorm**2) + np.dot(dB_dR, dB_dR) + np.dot(Bcyl, d2B_d2R)
        d2modB_dRdz = - np.dot(Bcyl, dB_dR)*np.dot(Bcyl, dB_dz) / (Bnorm**2) + np.dot(dB_dR, dB_dz) +\
            np.dot(Bcyl, d2B_dRdz)
        d2modB_d2z = - np.dot(Bcyl, dB_dz)**2 / (Bnorm**2) + np.dot(dB_dz, dB_dz) + np.dot(Bcyl, d2B_d2z)
        d2modB_d2R /= Bnorm
        d2modB_dRdz /= Bnorm
        d2modB_d2z /= Bnorm

        gradCyl_dmodB_dx = np.array([d2modB_d2R * costheta, -gradB_cyl[0] * sintheta / r,
                                    d2modB_dRdz * costheta])
        gradCyl_dmodB_dy = np.array([d2modB_d2R * sintheta, gradB_cyl[0] * costheta / r,
                                    d2modB_dRdz * sintheta])
        gradCyl_dmodB_dz = np.array([d2modB_dRdz, 0, d2modB_d2z])

        BHessian = np.zeros([3, 3])
        BHessian[0, :] = cyl2cart(gradCyl_dmodB_dx, x)
        BHessian[1, :] = cyl2cart(gradCyl_dmodB_dy, x)
        BHessian[2, :] = cyl2cart(gradCyl_dmodB_dz, x)

        return ABdBGuidingCenter(Adag_jac=0, A=A, Adag=Adag,
                                 B=B, Bgrad=B_grad, b=b, Bnorm=Bnorm, BHessian=BHessian, Bdag=Bdag)
