from typing import List, Tuple

from log import logger
from activation import IActivation


def forward_propogation(
    inputs: List[float],
    perceptron: List[List[List[float]]],
    activation: IActivation,
) -> Tuple[List[float], List[List[float]]]:
    """
    Прямое распространение сигнала через многослойный перцептрон.

    Args:
        inputs: входные значения (x)
        perceptron: весовые матрицы слоёв.
            perceptron[q][j][k] — вес от k-го нейрона слоя (q-1) к j-му нейрону слоя q.
        activation: функция активации

    Returns:
        (outputs, weighted_sums_output)
        outputs — выходы последнего слоя после активации
        weighted_sums_output — взвешенные суммы каждого слоя до активации,
            weighted_sums_output[q][j] = s_j^q
    """
    if not perceptron:
        raise RuntimeError("perceptron is empty")
    if len(inputs) != len(perceptron[0][0]):
        e_str = (
            f"incorrect size of inputs: expected {len(perceptron[0][0])}, got {len(inputs)}"
        )
        logger.error(e_str)
        raise RuntimeError(e_str)

    current_activations: List[float] = inputs
    weighted_sums_output: List[List[float]] = []

    for q in range(len(perceptron)):
        layer_sums: List[float] = []
        layer_activations: List[float] = []

        for j in range(len(perceptron[q])):
            s_j = 0.0
            for k in range(len(perceptron[q][j])):
                s_j += perceptron[q][j][k] * current_activations[k]
            layer_sums.append(s_j)
            layer_activations.append(activation.perform(s_j))

        weighted_sums_output.append(layer_sums)
        current_activations = layer_activations

    return current_activations, weighted_sums_output
        