
from abc import ABC, abstractmethod
from enum import Enum


class ActivationType(str, Enum):  # str для JSON-сериализации
    RELLU = "RELLU"
    SIGMOID = "SIGMOID"
    SOFTMAX = "SOFTMAX"




class IActivation(ABC):
    @abstractmethod
    def perform(self, value: float) -> float:
        """Применяет функцию активации к значению."""
        pass
    
    @abstractmethod
    def derivative(self, value: float) -> float:
        """
        Вычисляет производную функции активации.
        
        Args:
            value: взвешенная сумма (NET вход) до применения активации
            
        Returns:
            производная функции активации в точке value
        """
        pass

    @abstractmethod
    def get_type(self)-> str: pass

    