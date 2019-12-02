from integrators.implementations.rk4 import RK4


def explicitIntegratorFactory(integratorName, config):
    if integratorName == "RK4":
        return RK4(config)

    raise Exception("Invalid first guess integrator " + integratorName)
