
from typing import List


from nn_logic.training.itraining_algorithm import ITrainingAlgorithm
from nn_logic.loss import ILoss, LossType
# from nn_logic.training.activation import IActivation
from nn_logic.mathh import mv
from nn_logic.mathh.models import Perceptron
from nn_logic.training.activation import ActivationType


# from config import config
# from log import logger
# from exceptions.argument_exception import ArgumentException



class BackPropagation(ITrainingAlgorithm):

    p: Perceptron
    loss: ILoss
    learning_rate: float

    def __init__(self, loss: ILoss, learning_rate:float, 
                 perceptron: Perceptron):
        self.learning_rate = learning_rate
        self.p = perceptron
        self.loss = loss

    def training_iteration_calculate(self, inputs: List[float], outputs: List[float], expected: List[float],
                          weighted_sums_output: List[List[float]]):
        """
        Одна итерация обучения.
        
        Args:
            inputs: входные значения
            outputs: выходы нейронов выходного слоя (y_j^Q)
            expected: желаемые значения (d_j)
            weighted_sums_output: взвешенные суммы слоев до активации (s_j^Q). Необходимы что бы оптимизировать алгоритм
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
        num_layers = len(self.p.weights)
        local_errors: List[List[float]] = [[] for _ in range(num_layers)]


        # Если используется soft-max + cross-entropy то используется упрощенная формула без производных
        if(self.p.activations[-1].get_type() == ActivationType.SOFTMAX 
           and self.loss.get_type() == LossType.CROSS_ENTROPY):
            for j in range(len(outputs)):
                y_j = outputs[j]
                d_j = expected[j]
                s_j = weighted_sums_output[-1][j]

                local_error = (y_j - d_j)
                output_local_errors.append(local_error)

        # Редко используемая комбинация
        elif(self.p.activations[-1].get_type() == ActivationType.SOFTMAX 
           and self.loss.get_type() == LossType.MSE):
            for j in range(len(outputs)):
                y_j = outputs[j]
                d_j = expected[j]
                s_j = weighted_sums_output[-1][j]

                local_error = y_j*(y_j - d_j)
                output_local_errors.append(local_error)

        else:
        
            for j in range(len(outputs)):
                y_j = outputs[j]
                d_j = expected[j]
                s_j = weighted_sums_output[-1][j]
                
                
                derivative = self.p.activations[-1].derivative(s_j)
                local_error = (y_j - d_j) * derivative
                output_local_errors.append(local_error)


        local_errors[-1] += output_local_errors

        # ШАГ 4: Рассчитать локальные ошибки для всех скрытых слоев по формуле (1) (от последнего к первому)
        """
        Формула(1): W[q+1]^T * δ[q+1] ⊙ f'(s[q])  (Векторная форма)
        Описание формулы для понимания: 
        для каждого слоя от последнего (минуя выходной, последний слой)
        first-step: транспонированную матрицу весов  следующего слоя (тоесть слоя с номером q+1) умножить МАТРИЧНО на вектор локальных ошибок этого же (q+1) слоя.
        secod-step: Затем получившийся ВЕКТОРЫ умножить ПОЭЛЕМЕНТНО на ВЕКТОРЫ производных функции активации слоя q.

        """



        for q in range(num_layers - 2, -1, -1): # -2 так как выходной слой не учитывается в предыдущем шагеы
            first_step = mv.m_v_mtpc(mv.t_mtx(self.p.weights[q+1]), local_errors[q+1]) ## first-step

            derivatives = [self.p.activations[q].derivative(x) for x in weighted_sums_output[q]]
            q_local_errors = mv.v_v_elementwise(first_step, derivatives)

            local_errors[q] += q_local_errors
    


        # ШАГ 5: Рассчитать для всех слоев по формуле 3 или 4 значение изменения весов. (дельта корректировку)
        #  Обновление весов: Δw_ij^q = -η * δ_j^q * y_i^(q-1)
        # где Δw_ij^q - изменение веса связи от i-го нейрона слоя (q - 1) в j-му нейрону слоя q
        # η - скорость обучения, δ_j^q - локальная ошибка, 
        # y_i^(q-1) - вход с прошлого слоя


        adjustments: List[List[List[float]]] = \
        [[[0.0 for _ in range(len(self.p.weights[q][i]))] for i in range(len(self.p.weights[q]))]
        for q in range(num_layers)]

        for q in range(num_layers - 1, -1, -1): # а здесь мы итерируемся по всем слоям, включая выходной 
            for i in range(len(self.p.weights[q])): # номер строки нейрон - получатель
                for j in range(len(self.p.weights[q][i])): #номер столбца нейрон - источник
                    if q == 0:
                        y_prev = inputs[j]
                    else:
                        y_prev = self.p.activations[q-1].perform(weighted_sums_output[q-1][j])
                    adjust = (-self.learning_rate) * local_errors[q][i] * y_prev
                    adjustments[q][i][j] = adjust

        return adjustments
        

    def get_perceptron(self): return self.p

    def get_loss_function(self): return self.loss

    def get_losses(self): pass
        
    def get_output_loss(self): pass


    