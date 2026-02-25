from abc import ABC, abstractmethod
import math


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