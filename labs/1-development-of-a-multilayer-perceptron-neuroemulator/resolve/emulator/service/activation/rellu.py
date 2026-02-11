from .iactivation import IActivation


class ReLU(IActivation):
    
    def perform(self, value: float) -> float:
        return max(0.0, value)