# Guiding Field Configuration.
# Expose useful guiding center quantities starting from a EM field
# i.e. gradient of B, A_dagger etc (see GuidingField struct)

# ABdB field struct for guiding center system
from abc import ABC, abstractmethod
import collections
ABdBGuidingCenter = collections.namedtuple("ABdBGuidingCenter", "A Adag B Bgrad b Bnorm BHessian Bdag")
BdBGuidingCenter = collections.namedtuple("BdBGuidingCenter", "B Bgrad b Bnorm BHessian Bdag")


class AB_dB_FieldBuilder(ABC):
    @abstractmethod
    def compute(z):
        pass
