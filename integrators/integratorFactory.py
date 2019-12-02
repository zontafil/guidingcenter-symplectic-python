# factory for integrators.
# use this if you want to compose an integrator inside a class (i.e. particle)
#
# i.e. if you want to create a RK4 integrator:
#  rk = integratorFactory("RK4", config)
from integrators.implementations.rk4 import RK4
from integrators.implementations.explicit4_gaugeFree import SymplecticExplicit4_GaugeFree
from integrators.implementations.explicit3 import SymplecticExplicit3
from integrators.implementations.explicit3_gaugeFree import SymplecticExplicit3GaugeFree
from integrators.implementations.explicit4 import SymplecticExplicit4


def integratorFactory(integratorName, config):
    if integratorName == "RK4":
        return RK4(config)
    # elif integratorName == "VariationalMidpoint":
    #     return VariationalMidpoint(config)
    # elif integratorName == "SymplecticExplicit1":
    #     return SymplecticExplicit1(config)
    # elif integratorName == "SymplecticExplicit2":
    #     return SymplecticExplicit2(config)
    elif integratorName == "SymplecticExplicit3":
        return SymplecticExplicit3(config)
    elif integratorName == "SymplecticExplicit3GaugeFree":
        return SymplecticExplicit3GaugeFree(config)
    elif integratorName == "SymplecticExplicit4":
        return SymplecticExplicit4(config)
    # elif integratorName == "SymplecticImplicit1":
    #     return SymplecticImplicit1(config)
    # elif integratorName == "symplecticSemiexplicitQin":
    #     return SemiexplicitQin(config)
    # elif integratorName == "symplecticSemiexplicitQinRegularized":
    #     return SemiexplicitQinReg(config)
    elif integratorName == "SymplecticExplicit4_GaugeInvariant":
        return SymplecticExplicit4_GaugeFree(config)
    # elif integratorName == "SymplecticExplicit3_GaugeInvariant":
    #     return SymplecticExplicit3_GaugeFree(config)

    # else if (DIM==6){
    #     if (integratorName=="SymplecticImplicit3D") return new SymplecticImplicit3D<DIM>(config);
    # }

    raise Exception("Invalid integrator " + integratorName)


def explicitIntegratorFactory(integratorName, config):
    if integratorName == "RK4":
        return RK4(config)
    elif integratorName == "SymplecticExplicit4_GaugeInvariant":
        return SymplecticExplicit4_GaugeFree(config)

    raise Exception("Invalid first guess integrator " + integratorName)
