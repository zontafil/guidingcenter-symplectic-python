# tokamak field configuration. (case B)
# see  6.2.2 paragraph
from emFields.ABfields.emField import EMField
import numpy as np
from emFields.eqdskReader.eqdskReader import EqdskReader


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class GradShafranovSplineAB(EMField):
    def __init__(self, config):
        self.B0 = config.B0
        self.R0 = config.R0

        self.eqdsk = EqdskReader(config.eqdskFile, config.psi_degree, config.f_degree)

        print("EQDSK: range r: {} {}".format(self.eqdsk.r_min, self.eqdsk.r_max))
        print("EQDSK: range z: {} {}".format(self.eqdsk.z_min, self.eqdsk.z_max))
        print("EQDSK: range psi: {} {}".format(self.eqdsk.simag, self.eqdsk.sibry))

    def A(self, x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        z = x[2]
        psi = self.eqdsk.psi_spl(x=r, y=z)[0][0]

        Acyl = np.array([0, psi/r, np.log(r / self.R0)])
        return cyl2cart(Acyl, x)

    def B(self, x):
        R = np.sqrt(x[0]**2 + x[1]**2)
        Z = x[2]
        psi = self.eqdsk.psi_spl(x=R, y=Z)[0][0]
        dpsi_dR = self.eqdsk.psi_spl(x=R, y=Z, dx=1, dy=0, grid=True)[0][0]
        dpsi_dz = self.eqdsk.psi_spl(x=R, y=Z, dx=0, dy=1, grid=True)[0][0]
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

        curlA_cyl = np.zeros(3)
        curlA_cyl[0] = - dpsi_dz / R
        # curlA_cyl[1] = - 1 / R
        curlA_cyl[1] = - fpol / R
        curlA_cyl[2] = dpsi_dR / R

        return cyl2cart(curlA_cyl, x)
