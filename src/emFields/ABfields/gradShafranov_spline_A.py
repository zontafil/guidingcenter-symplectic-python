# tokamak field configuration. (case B)
# see  6.2.2 paragraph
from emFields.ABfields.emField import EMField
import numpy as np
from emFields.eqdskReader.eqdskReader import EqdskReader
import sys


def cyl2cart(v, x):
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    ret = np.zeros(3)
    ret[0] = (v[0]*x[0] - v[1]*x[1]) / r
    ret[1] = (v[0]*x[1] + v[1]*x[0]) / r
    ret[2] = v[2]
    return ret


class GradShafranovSplineA(EMField):
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

        if psi > max(self.eqdsk.sibry, self.eqdsk.simag) or psi < min(self.eqdsk.sibry, self.eqdsk.simag) \
           or z > self.eqdsk.sepmaxz or z < self.eqdsk.sepminz:
            print("WARNING: outside main plasma. psi: {}, Z: {}, R: {}".format(psi, z, r))

            sys.exit(0)

        Acyl = np.array([0, psi/r, -self.B0 * self.R0 * np.log(r / self.R0)])
        return cyl2cart(Acyl, x)

    def B(self, x):
        pass
