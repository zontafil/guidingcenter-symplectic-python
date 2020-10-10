from integrators.implementations.rk4 import RK4
from integrators.implementations.euler import Euler
from integrators.implementations.pauli_boris import PauliBoris


def explicitIntegratorFactory(integratorName, config):
    if integratorName == "RK4":
        return RK4(config)
    elif integratorName == "Euler":
        return Euler(config)
    elif integratorName == "PauliBoris":
        return PauliBoris(config)

    raise Exception("Invalid first guess integrator " + integratorName)
