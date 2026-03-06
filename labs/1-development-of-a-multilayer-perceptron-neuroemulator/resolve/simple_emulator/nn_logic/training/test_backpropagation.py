from typing import Any, List

from nn_logic.training.backpropagation import BackPropagation
from nn_logic.mathh.models import Perceptron
from nn_logic.loss import MSE
from nn_logic.training.activation import Rellu
from log import logger
from exceptions.test_exception import TestException

def test_bp_iteration():

    INPUTS = [0.5, 0.3]
    LEARNING_RATE = 0.1
    LAYERS_COUNT = 3
    WEIGHTS = \
    [
        [
            [  0.1, 0.4,],
            [0.8, -0.3],
            [0.2, 0.9],
        ],[
            [-0.2, 0.6, 0.5]
        ]
    ]

    OUTPUTS = [0.337]
    EXPECTED_OUTPUTS = [1.0]

    WEIGHTD_SUMS_OUTPUT = [
        [0.17, 0.31, 0.37],
        [0.337]
    ]

    EXPECTED_WEIGHTS_CHANGES = \
    [
        [
            [-0.00663, -0.00398],
            [0.01989, 0.01193],
            [00.01658, 0.00995],
        ],
        [
            [0.01127, 0.02055, 0.02453]
        ]
    ]


    p = Perceptron(weights=WEIGHTS, activations=[Rellu(), Rellu()], layers_count=LAYERS_COUNT)


    bp = BackPropagation(MSE(), LEARNING_RATE, p)

    result = bp.training_iteration_calculate(INPUTS, OUTPUTS, EXPECTED_OUTPUTS, WEIGHTD_SUMS_OUTPUT)

    errors:List[Any] = list()
    for q in range(len(EXPECTED_WEIGHTS_CHANGES)):
        for i in range(len(EXPECTED_WEIGHTS_CHANGES[q])):
            for j in range(len(EXPECTED_WEIGHTS_CHANGES[q][i])):
                w = EXPECTED_WEIGHTS_CHANGES[q][i][j]
                # if w != result[q][i][j]:
                if abs(w - result[q][i][j]) > 0.0001:
                    logger.error(f" Test error. incorrect weight  layer {q} | line {i} | column {j}")
                    errors.append({"position":(q, i, j), "expected":w , "received":result[q][i][j]})
    

    if(len(errors)):
        raise TestException(f" errors:  {errors}")
    else:
        logger.info("test complete!")