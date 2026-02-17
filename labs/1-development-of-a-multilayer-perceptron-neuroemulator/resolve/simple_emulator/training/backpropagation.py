
from typing import List


from training.itraining_algorithm import ITrainingAlgorithm
from .update_weights_algorithm import IUpdatingWeightsAlgorithm
from ..loss import ILoss
from ..activation import IActivation
from config import config



class BackPropagation(ITrainingAlgorithm):

    updw_alg: IUpdatingWeightsAlgorithm
    perceptrone: List[List[List[float]]]
    loss: ILoss
    activation: IActivation

    def __init__(self, loss: ILoss, update_weights_algorithm: IUpdatingWeightsAlgorithm, 
                 perceptrone: List[List[List[float]]], activation: IActivation):
        self.updw_alg = update_weights_algorithm
        self.perceptrone = perceptrone
        self.loss = loss
        self.activation = activation

    def training_iteration(self, inputs: List[float], outputs: List[float], expected: List[float],
                          weighted_sums_output: List[float]):
        """
        Одна итерация обучения.
        
        Args:
            inputs: входные значения
            outputs: выходы нейронов выходного слоя (y_j^Q)
            expected: желаемые значения (d_j)
            weighted_sums_output: взвешенные суммы выходного слоя до активации (s_j^Q). Необходимы что бы оптимизировать алгоритм
            и не вычислять значение сумматора нейрона до функции активации
        """

        # Вычисляем общую ошибку сети
        # loss_value: float = self.loss.perform(expected=expected, outputs=outputs)

        # ШАГ 3: Рассчитать локальные ошибки для выходного слоя
        # Формула (2): δ_j^Q = (y_j^Q - d_j) · f'(s_j^Q)
        output_local_errors: List[float] = []
        
        for j in range(len(outputs)):
            # y_j^Q - предсказание j-го нейрона выходного слоя
            y_j = outputs[j]
            # d_j - истинное (желаемое) значение
            d_j = expected[j]
            # s_j^Q - взвешенная сумма j-го нейрона выходного слоя
            s_j = weighted_sums_output[j]
            
            # f'(s_j^Q) - производная функции активации
            derivative = self.activation.derivative(s_j)
            
            # δ_j^Q = (y_j^Q - d_j) · f'(s_j^Q)
            local_error = (y_j - d_j) * derivative
            output_local_errors.append(local_error)

        prev_local_errors = output_local_errors

        # ШАГ 4: Рассчитать локальные ошибки для всех скрытых слоев по формуле (1) (от последнего к первому)
        # перебираем слои
        for i in range(config.LAYERS_COUNT, 1, -1):
            # перебираем нейроны - источник (веса нейронов предыдущего слоя)
            for j in range(len(self.perceptrone[i])):
                # перебираем локальные ошибки целевого слоя 
                for k in range(len(prev_local_errors)):
                    

        

    def get_perceptrone(self): return self.perceptrone

    def get_loos_function(self): return self.loss


    