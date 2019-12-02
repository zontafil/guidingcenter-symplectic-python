from emFields.eqdskReader.eqdskReader import EqdskReader
from emFields.AB_dBfields.AB_dBfield import AB_dB_FieldBuilder, BdBGuidingCenter
import numpy as np


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class SplineField_BdB(AB_dB_FieldBuilder):
    def __init__(self, config):
        self.eqdsk = EqdskReader(config.eqdskFile)

        print("EQDSK: range r: {} {}".format(self.eqdsk.r_min, self.eqdsk.r_max))
        print("EQDSK: range z: {} {}".format(self.eqdsk.z_min, self.eqdsk.z_max))
        print("EQDSK: range psi: {} {}".format(self.eqdsk.simag, self.eqdsk.sibry))

    def B_dB_cyl(self, R, Z):

        # interpolate psi and derivatives
        psi = self.eqdsk.psi_spl(x=R, y=Z)[0][0]
        dpsi_dR = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=0, grid=True)[0][0]
        dpsi_dz = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=1, grid=True)[0][0]
        d2psi_dR2 = self.eqdsk.psi_spl(x=R, y=Z, dx=2, dy=0, grid=True)[0][0]
        d2psi_dRdz = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=1, grid=True)[0][0]
        d2psi_dz2 = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=2, grid=True)[0][0]
        d3psi_d2Rdz = self.eqdsk.psi_spl(x=R, y=Z, dx=2, dy=1, grid=True)[0][0]
        d3psi_dRd2z = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=2, grid=True)[0][0]
        d3psi_d3z = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=3, grid=True)[0][0]
        d3psi_d3R = self.eqdsk.psi_spl(x=R, y=Z, dx=3, dy=0, grid=True)[0][0]

        # interpolate fpol and derivatives
        if psi < max(self.eqdsk.sibry, self.eqdsk.simag) and psi > min(self.eqdsk.sibry, self.eqdsk.simag) \
           and Z < self.eqdsk.sepmaxz and Z > self.eqdsk.sepminz:
            # most likely in the main plasma
            fpol = self.eqdsk.fpol_spl(psi)
            dfpol_dpsi = self.eqdsk.fpol_spl(psi, 1)
            d2fpol_d2psi = self.eqdsk.fpol_spl(psi, 2)
        else:
            print("WARNING: outside main plasma")
            # most likely outside the main plasma
            fpol = self.fpol[-1]
            dfpol_dpsi = 0.

        # evaluate the magnetic field
        BR = -dpsi_dz/R
        Bp = fpol/R
        Bz = dpsi_dR/R
        # evaluate the derivatives
        dBR_dR = dpsi_dz/(R**2)-d2psi_dRdz/R
        dBR_dp = 0.
        dBR_dz = -d2psi_dz2/R
        dBp_dR = -fpol/(R**2)+dfpol_dpsi*dpsi_dR/R
        dBp_dp = 0.
        dBp_dz = dfpol_dpsi*dpsi_dz/R
        dBz_dR = -dpsi_dR/(R**2) + d2psi_dR2/R
        dBz_dp = 0.
        dBz_dz = d2psi_dRdz/R

        d2BR_d2R = -2 * dpsi_dz / (R**3) + 2 * d2psi_dRdz / (R**2) - d3psi_d2Rdz / R
        d2BR_dRdz = d2psi_dz2 / (R**2) - d3psi_dRd2z / R
        d2BR_d2z = -d3psi_d3z / R
        # d2Bp_d2R = 2 * fpol / (R**3) - 2 * dfpol_dpsi * dpsi_dR / (R**2) + d2fpol_d2psi * d2psi_dR2 / R
        d2Bp_d2R = 2 * fpol / (R**3) - 2 * dfpol_dpsi * dpsi_dR / (R**2) + d2fpol_d2psi * dpsi_dR**2 / R +\
            dfpol_dpsi * d2psi_dR2 / R
        # d2Bp_dRdz = -dfpol_dpsi * dpsi_dz / (R**2) + d2fpol_d2psi * d2psi_dRdz / R
        d2Bp_dRdz = -dfpol_dpsi * dpsi_dz / (R**2) + d2fpol_d2psi * dpsi_dR * dpsi_dz / R + dfpol_dpsi * d2psi_dRdz / R
        # d2Bp_d2z = d2fpol_d2psi * d2psi_dz2 / R
        d2Bp_d2z = d2fpol_d2psi * dpsi_dz**2 / R + dfpol_dpsi * d2psi_dz2 / R
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

    def compute(self, z):
        x = z[:3]
        r = np.sqrt(x[0]**2 + x[1]**2)
        theta = np.arctan(x[1]/x[0])
        BdB = self.B_dB_cyl(r, x[2])

        Bcyl = np.array([BdB[0], BdB[1], BdB[2]])

        # build B, B and |B| (cartesian)
        B = cyl2cart(Bcyl, x)
        Bnorm = np.linalg.norm(B)
        b = B / Bnorm

        # build curl B and Bdag (cyl and cartesian)
        Bcurl_cyl = np.zeros(3)
        Bcurl_cyl[0] = - BdB[8]
        Bcurl_cyl[1] = BdB[5] - BdB[9]
        Bcurl_cyl[2] = Bcyl[1] / r + BdB[6]
        Bcurl = cyl2cart(Bcurl_cyl, x)
        Bdag = B + z[3] * Bcurl / Bnorm

        # build grad|B| (cyl and cartesian)
        dB_dR = np.array([BdB[3], BdB[6], BdB[9]])
        dB_dz = np.array([BdB[5], BdB[8], BdB[11]])
        gradB_cyl = np.zeros(3)
        gradB_cyl[0] = np.dot(Bcyl, dB_dR)
        gradB_cyl[1] = 0
        gradB_cyl[2] = np.dot(Bcyl, dB_dz)
        gradB_cyl /= Bnorm
        B_grad = cyl2cart(gradB_cyl, x)

        # compute B hessian
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

        return BdBGuidingCenter(B=B, Bgrad=B_grad, b=b, Bnorm=Bnorm, BHessian=BHessian, Bdag=Bdag)
