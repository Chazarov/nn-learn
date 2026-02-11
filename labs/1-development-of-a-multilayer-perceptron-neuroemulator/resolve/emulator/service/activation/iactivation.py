from abc import ABC, abstractmethod


class IActivation(ABC):

    @abstractmethod
    def perform(self, value: float) -> float: pass
