from abc import ABC, abstractmethod
from enum import Enum
import math
from typing import Any, Dict, List

from exceptions import UnexpectedBehaviourException
from log import logger


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


class Rellu(IActivation):
    def perform(self, value: float) -> float:
        """ReLU: f(x) = max(0, x)"""
        return max(0, value)
    
    def derivative(self, value: float) -> float:
        """
        Производная ReLU: f'(x) = 1 если x > 0, иначе 0
        
        Args:
            value: взвешенная сумма (NET вход)
        """
        return 1.0 if value > 0 else 0.0
    
    def get_type(self) -> str:
        return ActivationType.RELLU


class Sigmoid(IActivation):
    def perform(self, value: float) -> float:
        """Sigmoid: f(x) = 1 / (1 + e^(-x))"""
        return 1.0 / (1.0 + math.exp(-value))

    def derivative(self, value: float) -> float:
        """
        Производная Sigmoid: f'(x) = f(x) · (1 - f(x))

        Args:
            value: взвешенная сумма (NET вход)
        """
        s = self.perform(value)
        return s * (1.0 - s)
    
    def get_type(self) -> str:
        return ActivationType.SIGMOID
    

class ILayerBasedActivation(IActivation, ABC):

    @abstractmethod
    def set_layer_outputs(self, layer_weights_outputs: List[float] ) -> None:pass

class SoftMax(ILayerBasedActivation):

    _layer_sums_outputs: List[float]
    def perform(self, value:float) -> float:
       
        def exp(value:float)-> float:
            return math.e**value
        
        exp_sum = sum([exp(val) for val in self._layer_sums_outputs])

        return exp(value)/exp_sum
    
    def set_layer_outputs(self, layer_weights_outputs:List[float]):
        self._layer_sums_outputs = layer_weights_outputs

        
    def derivative(self, value: float) -> float:
        str_e = "For the soft-max activation, the differentiation algorithm (taking the derivative) is not implemented" \
        "Since this behaviour is not used in calculations. The soft-max function is used only on the output layer " \
        "whtch dose not require taking the derivative of the activation function" \
        "to calculate errors."
        logger.error(str_e)
        raise UnexpectedBehaviourException(str_e)
        
        
    def get_type(self) -> str:
        return ActivationType.SOFTMAX
        


        
        
        

    

ACTIVATIONS:Dict[str, Any] = {
    ActivationType.RELLU: Rellu,
    ActivationType.SIGMOID: Sigmoid
}