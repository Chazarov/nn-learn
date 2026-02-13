from abc import ABC, abstractmethod


class IActivation(ABC):
    @abstractmethod
    def perform(value:float):pass

class Rellu(IActivation):
    def perform(value:float):
        return max(0, value)