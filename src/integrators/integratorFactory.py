# factory for integrators.
# use this if you want to compose an integrator inside a class (i.e. particle)
#
# i.e. if you want to create a RK4 integrator:
#  rk = integratorFactory("RK4", config)
from integrators.implementations.rk4 import RK4
from integrators.implementations.pauli_boris import PauliBoris
from integrators.implementations.explicit4_gaugeFree import SymplecticExplicit4_GaugeFree
from integrators.implementations.explicit4_thirdOrder import SymplecticExplicit4_ThirdOrder
from integrators.implementations.explicit3 import SymplecticExplicit3
from integrators.implementations.explicit3_gaugeFree import SymplecticExplicit3GaugeFree
from integrators.implementations.explicit4 import SymplecticExplicit4
from integrators.implementations.variationalMidpoint import VariationalMidpoint
from integrators.implementations.implicit1 import SymplecticImplicit1
from integrators.implementations.euler import Euler
from integrators.implementations.degenerate3D import Degenerate3D
from integrators.implementations.PauliMidpointImplicit import PauliMidpoint


def integratorFactory(integratorName, config):
    if integratorName == "RK4":
        return RK4(config)
    elif integratorName == "Euler":
        return Euler(config)
    elif integratorName == "VariationalMidpoint":
        return VariationalMidpoint(config)
    elif integratorName == "SymplecticExplicit3":
        return SymplecticExplicit3(config)
    elif integratorName == "SymplecticExplicit3GaugeFree":
        return SymplecticExplicit3GaugeFree(config)
    elif integratorName == "SymplecticExplicit4":
        return SymplecticExplicit4(config)
    elif integratorName == "SymplecticImplicit1":
        return SymplecticImplicit1(config)
    elif integratorName == "SymplecticExplicit4_GaugeInvariant":
        return SymplecticExplicit4_GaugeFree(config)
    elif integratorName == "SymplecticExplicit4_ThirdOrder":
        return SymplecticExplicit4_ThirdOrder(config)
    elif integratorName == "PauliBoris":
        return PauliBoris(config)
    elif integratorName == "Degenerate3D":
        return Degenerate3D(config)
    elif integratorName == "PauliMidpoint":
        return PauliMidpoint(config)

    raise Exception("Invalid integrator " + integratorName)
