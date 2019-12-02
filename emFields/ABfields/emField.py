# EM field interface. Must implement vector potential and magnetic field
from abc import ABC, abstractmethod


class EMField(ABC):
    @abstractmethod
    def A(self, x):
        pass

    @abstractmethod
    def B(self, x):
        pass
