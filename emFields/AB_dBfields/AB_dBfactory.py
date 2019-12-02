from emFields.AB_dBfields.finiteDFromAB import FiniteDFromAB
from emFields.AB_dBfields.splineField_BdB import SplineField_BdB
from emFields.AB_dBfields.finiteDFromA import FiniteDFromA
from emFields.AB_dBfields.gradShafranov_ABdB import GradShafranov_ABdB


def AB_dB_FieldFactory(fieldName, config):
    if fieldName == "finiteDFromAB":
        return FiniteDFromAB(config)
    elif fieldName == "splineField":
        return SplineField_BdB(config)
    elif fieldName == "GradShafranovSplineABdB":
        return GradShafranov_ABdB(config)
    elif fieldName == "finiteDFromA":
        return FiniteDFromA(config)

    raise Exception("Invalid Guiding Field Algorithm (config->AB_dB_Algorithm). Choices: finiteDFromAB")
