

from abc import ABC, abstractmethod
from typing import List
from enum import Enum


class LossType(str, Enum):
    MSE = "MSE"
    CROSS_ENTROPY = "CROSS_ENTROPY"

    

class ILoss(ABC):
    @abstractmethod
    def perform(self, expected:List[float], outputs: List[float]) -> float: pass

    @abstractmethod
    def get_type(self) -> LossType: pass
