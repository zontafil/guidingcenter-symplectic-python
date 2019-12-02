# factory for EMfields.
# use this if you want to compose a field inside a class (i.e. guidingField)
# i.e. if you want to create a Tokamak field:
# field = EMFIeldFactory("Tokamak", config)
from emFields.ABfields.tokamak import Tokamak


def EMFieldFactory(fieldName, config):
    if fieldName == "Tokamak":
        return Tokamak(config)
    # elif fieldName == "ForceFree":
    #     return ForceFree(config)
    # elif fieldName == "TwoDimField":
    #     return TwoDimField(config)
    # elif fieldName == "TokamakElmfire":
    #     return TokamakElmfire(config)
    # elif fieldName == "splineFieldB":
    #     return SplineField_B(config)
    # elif fieldName == "splineShafranovA":
    #     return SplineShafranovA(config)

    raise Exception("Invalid EMField " + fieldName)
