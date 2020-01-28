from abc import ABC, abstractmethod


class System(ABC):
    @abstractmethod
    def f_eq_motion(self, z):
        pass
