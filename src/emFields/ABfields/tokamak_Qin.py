# tokamak field configuration.
# see  Qin (2009) and Qin (2020)

# notation:
# R - major radius: R = sqrt(x**2 + y**2)
# R0 == 1
# r**2 = (R - R0)**2 + z**2
from emFields.ABfields.emField import EMField
import numpy as np
from particleUtils import cyl2cart, cart2cyl


class TokamakQin(EMField):
    def __init__(self, config):
        self.B0 = config.B0

    def Acyl(self, x):
        return cart2cyl(self.A(x), x)

    def A(self, x):
        R = np.sqrt(x[0]**2 + x[1]**2)
        r = np.sqrt((R-1)**2 + x[2]**2)
        Acyl = np.zeros(3)
        Acyl[0] = x[2] / (R)
        Acyl[1] = r**2 / (2.*R)
        Acyl[2] = -np.log(R)
        Acyl *= 0.5 * self.B0

        return cyl2cart(Acyl, x)

    def B(self, x):
        R = np.sqrt(x[0]**2 + x[1]**2)
        r = np.sqrt((R-1)**2 + x[2]**2)
        Bcyl = np.zeros(3)
        Bcyl[1] = self.B0 / R

        Bpol = self.B0 * r / (2.*R)
        Bcyl[0] = - Bpol*x[2] / r
        Bcyl[2] = Bpol*(R-1) / r

        return cyl2cart(Bcyl, x)
