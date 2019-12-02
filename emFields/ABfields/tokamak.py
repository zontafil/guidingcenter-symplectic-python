# tokamak field configuration. (case B)
# see  6.2.2 paragraph
from emFields.ABfields.emField import EMField
import numpy as np


class Tokamak(EMField):
    def __init__(self, config):
        self.B0 = config.B0
        self.R0 = config.R0
        self.q = config.q

    def A(self, x):
        ret = np.zeros(3)
        ret[0] = 0
        ret[1] = -self.B0*self.R0*np.log((self.R0+x[0])/self.R0)
        ret[2] = self.B0/(2.*self.q*(self.R0+x[0]))*(2.*self.R0*(self.R0+x[0])*np.log((self.R0+x[0])/self.R0)
                                                     - 2.*self.R0*x[0] - 2.*x[0]*x[0] - x[1]*x[1])
        return ret

    def B(self, x):
        ret = np.zeros(3)
        ret[0] = -self.B0*x[1]/(self.q*(self.R0+x[0]))
        ret[1] = -((self.B0 * (self.R0 * x[0] * (-2.) - 2.*x[0]*x[0]
                               + x[1]*x[1]))/(2. * self.q * (self.R0 + x[0])*(self.R0 + x[0])))
        ret[2] = -self.B0*self.R0/(self.R0+x[0])
        return ret
