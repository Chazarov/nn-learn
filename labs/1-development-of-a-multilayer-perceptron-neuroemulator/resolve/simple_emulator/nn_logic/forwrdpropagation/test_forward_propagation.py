from typing import Any, List

from nn_logic.forwrdpropagation.forward_propagation import forward_propogation
from nn_logic.activation import Rellu
from exceptions.test_exception import TestException
from log import logger


def test_forward_propagation():
    """
    Checking forward propagation outputs and weighted sums.
    Based on the example from neuro_hint2.md (network 2 -> 3 -> 1, ReLU).
    """
    INPUTS = [0.5, 0.3]

    PERCEPTRONE = \
    [
        [
            [0.1, 0.4],
            [0.8, -0.3],
            [0.2, 0.9],
        ],[
            [-0.2, 0.6, 0.5]
        ]
    ]

    EXPECTED_OUTPUTS = [0.337]

    EXPECTED_WEIGHTED_SUMS = [
        [0.17, 0.31, 0.37],
        [0.337]
    ]


    outputs, weighted_sums = forward_propogation(INPUTS, PERCEPTRONE, Rellu())


    errors: List[Any] = list()

    for q in range(len(EXPECTED_WEIGHTED_SUMS)):
        for j in range(len(EXPECTED_WEIGHTED_SUMS[q])):
            expected = EXPECTED_WEIGHTED_SUMS[q][j]
            actual = weighted_sums[q][j]
            if abs(expected - actual) > 0.0001:
                logger.error(f"Test error. Incorrect weighted sum: layer {q} | neuron {j} | expected {expected} | got {actual}")
                errors.append(("weighted_sum", q, j))

    for j in range(len(EXPECTED_OUTPUTS)):
        expected = EXPECTED_OUTPUTS[j]
        actual = outputs[j]
        if abs(expected - actual) > 0.0001:
            logger.error(f"Test error. Incorrect output: neuron {j} | expected {expected} | got {actual}")
            errors.append(("output", j))


    if len(errors):
        raise TestException(f" errors: {errors}")
    else:
        logger.info("test complete!")
