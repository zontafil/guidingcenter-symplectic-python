# factory for EMfields.
# use this if you want to compose a field inside a class (i.e. guidingField)
# i.e. if you want to create a Tokamak field:
# field = EMFIeldFactory("Tokamak", config)
from emFields.ABfields.tokamak import Tokamak
from emFields.ABfields.gradShafranov_spline_A import GradShafranovSplineA
from emFields.ABfields.gradShafranov_spline_AB import GradShafranovSplineAB
from emFields.ABfields.gradShafranov_analytic_AB import GradShafranov_analytic_AB
from emFields.AB_dBfields.gradShafranov_analytic_ITER import ITERfield


def EMFieldFactory(fieldName, config):
    if fieldName == "Tokamak":
        return Tokamak(config)
    if fieldName == "GradShafranovSplineA":
        return GradShafranovSplineA(config)
    if fieldName == "GradShafranovSplineAB":
        return GradShafranovSplineAB(config)
    if fieldName == "GradShafranovAnalyticAB":
        return GradShafranov_analytic_AB(config)
    if fieldName == "ITERfield":
        return ITERfield(config)
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
