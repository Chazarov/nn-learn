from typing import List, Tuple

from log import logger
from nn_logic.models.activation import ActivationType
from nn_logic.training.activation.activation import SoftMax
from nn_logic.mathh.models import Perceptron



def forward_propagation(
    inputs: List[float],
    perceptron: Perceptron,
) -> Tuple[List[float], List[List[float]]]:
    """
    Прямое распространение сигнала через многослойный перцептрон.

    Args:
        inputs: входные значения (x)
        perceptron: весовые матрицы слоёв.
            perceptron[q][j][k] — вес от k-го нейрона слоя (q-1) к j-му нейрону слоя q.

    Returns:
        (outputs, weighted_sums_output)
        outputs — выходы последнего слоя после активации
        weighted_sums_output — взвешенные суммы каждого слоя до активации,
            weighted_sums_output[q][j] = s_j^q
    """
    
    p = perceptron

    if len(inputs) != len(p.weights[0][0]):
        e_str = (
            f"incorrect size of inputs: expected {len(p.weights[0][0])}, got {len(inputs)}"
        )
        logger.error(e_str)
        raise RuntimeError(e_str)

    current_activations: List[float] = inputs
    weighted_sums_output: List[List[float]] = []



    for q in range(p.layers_count - 1):
        layer_sums: List[float] = []
        layer_activations: List[float] = []


        for j in range(len(p.weights[q])):
            s_j = 0.0
            for k in range(len(p.weights[q][j])):
                s_j += p.weights[q][j][k] * current_activations[k]
            layer_sums.append(s_j)

            if p.activations[q].get_type() != ActivationType.SOFTMAX:
                layer_activations.append(p.activations[q].perform(s_j))

        if p.activations[q].get_type() == ActivationType.SOFTMAX:
            softmax_act: SoftMax = p.activations[q]  # type: ignore[assignment]
            softmax_act.set_layer_outputs(layer_sums)
            layer_activations = [p.activations[q].perform(s) for s in layer_sums]

        weighted_sums_output.append(layer_sums)
        current_activations = layer_activations

    return current_activations, weighted_sums_output
        