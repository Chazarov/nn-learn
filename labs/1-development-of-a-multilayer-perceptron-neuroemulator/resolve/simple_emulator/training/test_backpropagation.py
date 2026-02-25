from typing import Any, List

from .backpropagation import BackPropagation
from ..activation import Rellu
from ..loss import MSE
from ..exceptions.test_exception import TestException
from ..log import logger



def test_iteration():
    """
    Checking the calculation of weight changes after the 
    backpropagation training iteration
    """
    LEARNING_RATE = 0.1
    INPUTS = [0.5, 0.3]

    OUTPUTS = [0.337]
    EXPECTED_OUTPUTS = [1.0]

    PERCEPTRONE = \
    [
        [
            [  0.1, 0.4,],
            [0.8, -0.3],
            [0.2, 0.9],
        ],[
            [-0.2, 0.6, 0.5]
        ]
    ]

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



    backprop = BackPropagation(MSE(), LEARNING_RATE, PERCEPTRONE, Rellu())


    result = backprop.training_iteration_calculate(INPUTS, OUTPUTS, EXPECTED_OUTPUTS, WEIGHTD_SUMS_OUTPUT)

    errors:List[Any] = list()
    for q in range(len(EXPECTED_WEIGHTS_CHANGES)):
        for i in range(len(EXPECTED_WEIGHTS_CHANGES[q])):
            for j in range(len(EXPECTED_WEIGHTS_CHANGES[q][i])):
                w = EXPECTED_WEIGHTS_CHANGES[q][i][j]
                if abs(w - result[q][i][j]) > 0.0001:
                    logger.error(f" Test error. incorrect weight  layer {q} | line {i} | column {j}")
                    errors.append((q, i, j))
    

    if(len(errors)):
        raise TestException(f" errors:  {errors}")
    else:
        logger.info("test complete!")





