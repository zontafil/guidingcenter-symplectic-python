# Guiding Field Configuration.
# Expose useful guiding center quantities starting from a EM field
# i.e. gradient of B, A_dagger etc (see GuidingField struct)

# ABdB field struct for guiding center system
from abc import ABC, abstractmethod
import collections
ABdBGuidingCenter = collections.namedtuple("ABdBGuidingCenter", "A Adag Adag_jac B Bgrad b Bnorm BHessian Bdag")
# ABdBGuidingCenter = collections.namedtuple("ABdBGuidingCenter", "A Adag Adag_jac B Bgrad b Bnorm BHessian Bdag d2modB_d2R d2modB_dRdz d2modB_d2z gradB_cyl gradCyl_dmodB_dx gradCyl_dmodB_dz gradCyl_dmodB_dy")
BdBGuidingCenter = collections.namedtuple("BdBGuidingCenter", "B Bgrad b Bnorm BHessian Bdag")


class AB_dB_FieldBuilder(ABC):
    @abstractmethod
    def compute(z):
        pass
