from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List

from exceptions import ArgumentException
from log import logger



class LossType(str, Enum):
    MSE = "MSE"
    CROSS_ENTROPY = "CROSS_ENTROPY"

    

class ILoss(ABC):
    @abstractmethod
    def perform(self, expected:List[float], outputs: List[float]) -> float: pass

    @abstractmethod
    def get_type(self) -> LossType: pass


class MSE(ILoss):
    """

    Mean Squared Error
    
    """
    def perform(self, expected:List[float], outputs: List[float]) -> float: 
        if(len(expected) != len(outputs)):
            e_str = "the size of arrays does not math"
            logger.error(e_str)
            raise ArgumentException(e_str)
        
        return sum([(expected[i] - outputs[i])**2 for i in range(len(outputs))])
    
    def get_type(self) -> LossType:
        return LossType.MSE
    

class CrossEntropy(ILoss):
    """
    Cross-Entropy Loss

    L = -sum(y_i * log(p_i))

    Используется совместно с softmax на выходном слое.
    expected — one-hot вектор (или распределение вероятностей).
    outputs  — предсказанные вероятности (сумма должна быть ~1).
    """

    _EPS = 1e-12  # клип для защиты от log(0)

    def perform(self, expected: List[float], outputs: List[float]) -> float:
        if len(expected) != len(outputs):
            e_str = "the size of arrays does not match"
            logger.error(e_str)
            raise ArgumentException(e_str)

        return -sum(
            expected[i] * math.log(max(outputs[i], self._EPS)) #type: ignore
            for i in range(len(outputs))
        )
    
    def get_type(self) -> LossType:
        return LossType.CROSS_ENTROPY
    


LOSSES:Dict[str, Any] = {
    LossType.MSE: MSE,
    LossType.CROSS_ENTROPY: CrossEntropy
}