from emFields.eqdskReader.eqdskReader import EqdskReader
from emFields.AB_dBfields.AB_dBfield import AB_dB_FieldBuilder, ABdBGuidingCenter
import numpy as np


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class GradShafranov_ABdB(AB_dB_FieldBuilder):
    def __init__(self, config):
        self.eqdsk = EqdskReader(config.eqdskFile)
        self.R0 = config.R0
        self.hx = config.hx

        print("EQDSK: range r: {} {}".format(self.eqdsk.r_min, self.eqdsk.r_max))
        print("EQDSK: range z: {} {}".format(self.eqdsk.z_min, self.eqdsk.z_max))
        print("EQDSK: range psi: {} {}".format(self.eqdsk.simag, self.eqdsk.sibry))

    def B_dB_cyl(self, R, Z):

        # interpolate psi and derivatives
        dpsi_dR = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=0, grid=True)[0][0]
        dpsi_dz = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=1, grid=True)[0][0]
        d2psi_dR2 = self.eqdsk.psi_spl(x=R, y=Z, dx=2, dy=0, grid=True)[0][0]
        d2psi_dRdz = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=1, grid=True)[0][0]
        d2psi_dz2 = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=2, grid=True)[0][0]
        d3psi_d2Rdz = self.eqdsk.psi_spl(x=R, y=Z, dx=2, dy=1, grid=True)[0][0]
        d3psi_dRd2z = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=2, grid=True)[0][0]
        d3psi_d3z = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=3, grid=True)[0][0]
        d3psi_d3R = self.eqdsk.psi_spl(x=R, y=Z, dx=3, dy=0, grid=True)[0][0]

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

    def A(self, x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        z = x[2]
        theta = np.arctan(x[1]/x[0])

        psi = self.eqdsk.psi_spl(x=r, y=z)[0][0]
        dpsi_dR = self.eqdsk.psi_spl(x=r, y=z, dx=1, dy=0, grid=True)[0][0]
        dpsi_dz = self.eqdsk.psi_spl(x=r, y=z, dx=0, dy=1, grid=True)[0][0]

        Acyl = np.array([0, psi/r, np.log(r / self.R0)])
        A = cyl2cart(Acyl, x)

        # compute Ajac
        dAx_dr = psi / r**2 * np.sin(theta) - dpsi_dR / r * np.sin(theta)
        dAx_dp = - psi / r**2 * np.cos(theta)
        dAx_dz = - dpsi_dz / r * np.sin(theta)
        dAy_dr = - psi / r**2 * np.cos(theta) + dpsi_dR / r * np.cos(theta)
        dAy_dp = - psi / r**2 * np.sin(theta)
        dAy_dz = dpsi_dz / r * np.cos(theta)
        dAz_dr = 1 / r
        dAz_dp = 0
        dAz_dz = 0
        Ajac = np.zeros([3, 3])
        Ajac[0, :] = cyl2cart(np.array([dAx_dr, dAx_dp, dAx_dz]), x)
        Ajac[1, :] = cyl2cart(np.array([dAy_dr, dAy_dp, dAy_dz]), x)
        Ajac[2, :] = cyl2cart(np.array([dAz_dr, dAz_dp, dAz_dz]), x)
        return [A, Ajac]

    def compute(self, z):
        x = z[:3]
        r = np.sqrt(x[0]**2 + x[1]**2)
        u = z[3]
        theta = np.arctan(x[1]/x[0])

        BdB = self.B_dB_cyl(r, x[2])
        Bcyl = np.array([BdB[0], BdB[1], BdB[2]])

        # build B, B and |B| (cartesian)
        B = cyl2cart(Bcyl, x)
        Bnorm = np.linalg.norm(B)
        b = B / Bnorm

        A_Ajac = self.A(x)
        A = A_Ajac[0]
        Ajac = A_Ajac[1]
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
        dBx_dr = BdB[3] * np.cos(theta) - BdB[6] * np.sin(theta)
        dBx_dp = - BdB[0] * np.sin(theta) / r - BdB[1] * np.cos(theta) / r
        dBx_dz = BdB[5] * np.cos(theta) - BdB[8] * np.sin(theta) / r
        dBy_dr = BdB[3] * np.sin(theta) + BdB[6] * np.cos(theta)
        dBy_dp = + BdB[0] * np.cos(theta) / r - BdB[1] * np.sin(theta) / r
        dBy_dz = BdB[5] * np.sin(theta) + BdB[8] * np.cos(theta) / r
        dBz_dr = BdB[9]
        dBz_dp = 0
        dBz_dz = BdB[11]
        Bjac = np.zeros([3, 3])
        Bjac[0, :] = cyl2cart(np.array([dBx_dr, dBx_dp, dBx_dz]), x)
        Bjac[1, :] = cyl2cart(np.array([dBy_dr, dBy_dp, dBy_dz]), x)
        Bjac[2, :] = cyl2cart(np.array([dBz_dr, dBz_dp, dBz_dz]), x)

        Adag_jac = Ajac + u * Bjac / Bnorm
        Adag_jac[0, :] += u * B[0] * np.transpose(grad1_B)
        Adag_jac[1, :] += u * B[1] * np.transpose(grad1_B)
        Adag_jac[2, :] += u * B[2] * np.transpose(grad1_B)
        # Bjac_comp = np.zeros([3, 3])

        # Adag_jac = np.zeros([3, 3])
        # for j in range(3):
        #     x0 = np.array(x)
        #     x1 = np.array(x)
        #     x0[j] -= self.hx
        #     x1[j] += self.hx

        #     r0 = np.sqrt(x0[0]**2 + x0[1]**2)
        #     B0dB = self.B_dB_cyl(r0, x0[2])
        #     B0cyl = np.array([B0dB[0], B0dB[1], B0dB[2]])
        #     B0 = cyl2cart(B0cyl, x0)
        #     r1 = np.sqrt(x1[0]**2 + x1[1]**2)
        #     B1dB = self.B_dB_cyl(r1, x1[2])
        #     B1cyl = np.array([B1dB[0], B1dB[1], B1dB[2]])
        #     B1 = cyl2cart(B1cyl, x1)

        #     B1norm = np.linalg.norm(B1)
        #     B0norm = np.linalg.norm(B0)

        #     # Bjac_comp[:, j] = 0.5*(B1 - B0) / self.hx
        #     # Adag_jac[:, j] = Ajac[:, j] + 0.5*(u*B1/B1norm - u*B0/B0norm) / self.hx

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

        gradCyl_dmodB_dx = np.array([d2modB_d2R * np.cos(theta), -gradB_cyl[0] * np.sin(theta) / r,
                                    d2modB_dRdz * np.cos(theta)])
        gradCyl_dmodB_dy = np.array([d2modB_d2R * np.sin(theta), gradB_cyl[0] * np.cos(theta) / r,
                                    d2modB_dRdz * np.sin(theta)])
        gradCyl_dmodB_dz = np.array([d2modB_dRdz, 0, d2modB_d2z])

        BHessian = np.zeros([3, 3])
        BHessian[0, :] = cyl2cart(gradCyl_dmodB_dx, x)
        BHessian[1, :] = cyl2cart(gradCyl_dmodB_dy, x)
        BHessian[2, :] = cyl2cart(gradCyl_dmodB_dz, x)

        return ABdBGuidingCenter(Adag_jac=Adag_jac, A=A, Adag=Adag,
                                 B=B, Bgrad=B_grad, b=b, Bnorm=Bnorm, BHessian=BHessian, Bdag=Bdag)
