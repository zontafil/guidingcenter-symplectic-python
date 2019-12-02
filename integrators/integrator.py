from systems.systemFactory import systemFactory


class Integrator():

    def __init__(self, config):
        self.config = config
        self.system = systemFactory(config.system, config)
