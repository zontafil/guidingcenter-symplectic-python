from emFields.AB_dBfields.finiteDFromAB import FiniteDFromAB
from emFields.AB_dBfields.splineField_BdB import SplineField_BdB
from emFields.AB_dBfields.finiteDFromA import FiniteDFromA


def AB_dB_FieldFactory(fieldName, config):
    if fieldName == "finiteDFromAB":
        return FiniteDFromAB(config)
    # if fieldName == "finiteDFromA":
    #     return FiniteDFromA(config)
    elif fieldName == "splineField":
        return SplineField_BdB(config)
    elif fieldName == "finiteDFromA":
        return FiniteDFromA(config)

    raise Exception("Invalid Guiding Field Algorithm (config->AB_dB_Algorithm). Choices: finiteDFromAB")
