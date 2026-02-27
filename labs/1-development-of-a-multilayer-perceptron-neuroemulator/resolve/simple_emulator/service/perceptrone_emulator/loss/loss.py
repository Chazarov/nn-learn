from abc import ABC, abstractmethod
from typing import List

from log import logger


class ILoss(ABC):
    @abstractmethod
    def perform(self, expected:List[float], outputs: List[float]) -> float: pass


class MSE(ILoss):
    """
    Docstring for MSE

    Mean Squared Error
    
    """
    def perform(self, expected:List[float], outputs: List[float]) -> float: 
        if(len(expected) != len(outputs)):
            e_str = "the size of arrays does not math"
            logger.error(e_str)
            raise RuntimeError(e_str)
        
        return sum([(expected[i] - outputs[i])**2 for i in range(len(outputs))])