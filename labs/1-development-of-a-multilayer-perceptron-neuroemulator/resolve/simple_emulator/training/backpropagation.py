
from typing import List


from training.itraining_algorithm import ITrainingAlgorithm
from .update_weights_algorithm import IUpdatingWeightsAlgorithm
from ..loss import ILoss
from ..activation import IActivation
from config import config
from log import logger
from mathh import mv
from exceptions.argument_exception import ArgumentException



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
        """
        Вычисляем общую ошибку сети
        loss_value: float = self.loss.perform(expected=expected, outputs=outputs)
        """
        

        # ШАГ 3: Рассчитать локальные ошибки для выходного слоя
        """
        Формула (2): δ_j^Q = (y_j^Q - d_j) · f'(s_j^Q) (скалярная)

        y_j^Q - предсказание j-го нейрона выходного слоя
        d_j - истинное (желаемое) значение
        s_j^Q - взвешенная сумма j-го нейрона выходного слоя
        f'(s_j^Q) - производная функции активации
        """
        
        output_local_errors: List[float] = []
        
        for j in range(len(outputs)):
            
            y_j = outputs[j]
            
            d_j = expected[j]
            
            s_j = weighted_sums_output[j]
            
            
            derivative = self.activation.derivative(s_j)
            local_error = (y_j - d_j) * derivative
            output_local_errors.append(local_error)

        prev_local_errors = output_local_errors

        # ШАГ 4: Рассчитать локальные ошибки для всех скрытых слоев по формуле (1) (от последнего к первому)
        """
        Формула(1): W[q+1]^T * δ[q+1] ⊙ f'(s[q])  (Векторная форма)
        Описание формулы для понимания: 
        для каждого слоя от последнего (минуя выходной, последний слой)
        first-step: транспонированную матрицу весов  следующего слоя (тоесть слоя с номером q+1) умножить МАТРИЧНО на вектор локальных ошибок этого же (q+1) слоя.
        secod-step: Затем получившийся ВЕКТОРЫ умножить ПОЭЛЕМЕНТНО на ВЕКТОРЫ производных функции активации слоя q.

        """


        num_layers = len(self.perceptrone)  # общее количество слоев (включая выходной)
        q_local_errors = None

        for q in range(num_layers - 2, -1, -1):
            first_step = mv.m_v_mtpc(mv.t_mtx(self.perceptrone[q+1]), prev_local_errors) ## first-step
            q_local_errors = mv.v_v_elementwise(first_step, ???) ## second-step TODO: умножить ПОЭЛЕМЕНТНО на ВЕКТОРЫ производных функции активации слоя q.

            prev_local_errors = q_local_errors
    

        

    def get_perceptrone(self): return self.perceptrone

    def get_loos_function(self): return self.loss


    