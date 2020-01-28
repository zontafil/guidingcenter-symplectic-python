# factory for systems.
# use this if you want to compose a system inside a class (i.e. particle)
# i.e. if you want to create a GuidingCenter system:
#  gc = systemFactory("GuidingCenter", config)
from systems.guidingCenter import GuidingCenter


def systemFactory(systemName, config):
    if systemName == "GuidingCenter":
        return GuidingCenter(config)
    # elif systemName == "GuidingCenterRegularized":
    #     return GuidingCenterRegularized(config)
    raise Exception("Invalid system " + systemName)
