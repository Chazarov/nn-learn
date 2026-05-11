

from abc import ABC, abstractmethod
from typing import List

class ITrainingAlgorithm(ABC):
    
    @abstractmethod
    def training_iteration_calculate(self, inputs: List[float], outputs: List[float], expected: List[float], weighted_sums_output: List[List[float]]) -> List[List[List[float]]]: pass

    @abstractmethod
    def get_losses(self): pass
        
    @abstractmethod
    def get_output_loss(self): pass
