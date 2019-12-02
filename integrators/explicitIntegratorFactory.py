from integrators.implementations.rk4 import RK4
from integrators.implementations.euler import Euler


def explicitIntegratorFactory(integratorName, config):
    if integratorName == "RK4":
        return RK4(config)
    elif integratorName == "Euler":
        return Euler(config)

    raise Exception("Invalid first guess integrator " + integratorName)
