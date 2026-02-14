
from abc import ABC, abstractmethod
from typing import List


class IUpdatingWeightsAlgorithm(ABC):
    """
    Интерфейс для алгоритмов обновления весов нейронной сети.
    """
    @abstractmethod
    def perform(self, 
                local_errors: List[List[float]], 
                layer_outputs: List[List[float]]) -> List[List[List[float]]]:
        """
        Обновляет веса перцептрона на основе локальных ошибок и выходов слоев.
        
        Args:
            perceptron: веса нейронной сети [слой][нейрон][вес]
            local_errors: локальные ошибки для каждого слоя и нейрона (δ_j^q)
            layer_outputs: выходы нейронов для каждого слоя (y_i^{q-1})
            learning_rate: скорость обучения (η)
            
        Returns:
            обновленный perceptron с новыми весами
        """
        pass

    @abstractmethod
    def get_learning_rate(): pass

    @abstractmethod
    def get_perceptrone(): pass


class UpdatingWeightsAlgorithmBase(IUpdatingWeightsAlgorithm):
    """
    Базовый алгоритм обновления весов без момента.
    Формула (3): Δw_ij^q = -η · δ_j^q · y_i^{q-1}
    """
    def __init__(self, learning_rate:float, perceptrone: List[List[List[float]]]):
        self.learning_rate = learning_rate
        self.perceptrone = perceptrone
    
    def perform(self, 
                local_errors: List[List[float]], 
                layer_outputs: List[List[float]]) -> List[List[List[float]]]:
        """
        Обновляет веса перцептрона по базовой формуле без момента.
        
        Args:
            perceptron: веса нейронной сети [слой][нейрон][вес]
            local_errors: локальные ошибки для каждого слоя и нейрона (δ_j^q)
            layer_outputs: выходы нейронов для каждого слоя (y_i^{q-1})
            learning_rate: скорость обучения (η)
            
        Returns:
            обновленный perceptron с новыми весами
        """
        # Создаем копию perceptron для обновления
        updated_perceptron: List[List[List[float]]] = [[[weight for weight in neuron] for neuron in layer] for layer in self.perceptrone]
        
        # Проходим по всем слоям
        for q in range(len(self.perceptrone)):
            # Проходим по всем нейронам текущего слоя
            for j in range(len(self.perceptrone[q])):
                # Проходим по всем весам текущего нейрона (связи с предыдущим слоем)
                for i in range(len(self.perceptrone[q][j])):
                    # Формула (3): Δw_ij^q = -η · δ_j^q · y_i^{q-1}
                    delta_weight = -self.learning_rate * local_errors[q][j] * layer_outputs[q][i]
                    
                    # Обновляем вес: w_ij^q(новый) = w_ij^q(старый) + Δw_ij^q
                    updated_perceptron[q][j][i] += delta_weight
        
        return updated_perceptron


class UpdatingWeightsAlgorithmWithMoment(IUpdatingWeightsAlgorithm):
    """
    Алгоритм обновления весов с моментом (momentum).
    Формула (4): Δw_ij^q(t) = -η · (μ · Δw_ij^q(t-1) + (1-μ) · δ_j^q · y_i^{q-1})
    
    Момент ускоряет обучение, сглаживает колебания градиента, 
    помогает выбраться из локальных минимумов.
    """
    
    def __init__(self, momentum: float, learning_rate: float, perceptrone: List[List[List[float]]]):
        """
        Инициализирует алгоритм с коэффициентом момента.
        
        Args:
            momentum: коэффициент момента (μ), обычно 0.9
        """
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.perceptrone = perceptrone
        # Храним предыдущие изменения весов для момента
        self.previous_weight_changes: List[List[List[float]]] = []
    
    def perform(self,  
                local_errors: List[List[float]], 
                layer_outputs: List[List[float]]) -> List[List[List[float]]]:
        """
        Обновляет веса перцептрона с учетом момента.
        
        Args:
            perceptron: веса нейронной сети [слой][нейрон][вес]
            local_errors: локальные ошибки для каждого слоя и нейрона (δ_j^q)
            layer_outputs: выходы нейронов для каждого слоя (y_i^{q-1})
            learning_rate: скорость обучения (η)
            
        Returns:
            обновленный perceptron с новыми весами
        """
        # Создаем копию perceptron для обновления
        updated_perceptron = [[[weight for weight in neuron] for neuron in layer] for layer in self.perceptrone]
        
        # Инициализируем previous_weight_changes при первом вызове
        if not self.previous_weight_changes:
            self.previous_weight_changes = [[[0.0 for _ in neuron] for neuron in layer] for layer in self.perceptrone]
        
        # Создаем структуру для хранения текущих изменений
        current_weight_changes = [[[0.0 for _ in neuron] for neuron in layer] for layer in self.perceptrone]
        
        # Проходим по всем слоям
        for q in range(len(self.perceptrone)):
            # Проходим по всем нейронам текущего слоя
            for j in range(len(self.perceptrone[q])):
                # Проходим по всем весам текущего нейрона (связи с предыдущим слоем)
                for i in range(len(self.perceptrone[q][j])):
                    # Формула (4): Δw_ij^q(t) = -η · (μ · Δw_ij^q(t-1) + (1-μ) · δ_j^q · y_i^{q-1})
                    delta_weight = -self.learning_rate * (
                        self.momentum * self.previous_weight_changes[q][j][i] + 
                        (1 - self.momentum) * local_errors[q][j] * layer_outputs[q][i]
                    )
                    
                    # Сохраняем текущее изменение для следующей итерации
                    current_weight_changes[q][j][i] = delta_weight
                    
                    # Обновляем вес: w_ij^q(новый) = w_ij^q(старый) + Δw_ij^q(t)
                    updated_perceptron[q][j][i] += delta_weight
        
        # Сохраняем текущие изменения как предыдущие для следующей итерации
        self.previous_weight_changes = current_weight_changes
        
        return updated_perceptron
    



