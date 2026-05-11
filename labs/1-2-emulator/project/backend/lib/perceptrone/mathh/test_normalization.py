from typing import Any, List

from nn_logic.mathh.models import Sample
from nn_logic.mathh.mv import (
    min_max_function,
    min_max_signs_normalize,
    min_max_samples_normalaize,
)

from log import logger
from exceptions.test_exception import TestException


TOLERANCE = 0.0001

# ─────────────────────────────────────────────
#  Данные для тестов
# ─────────────────────────────────────────────
#
#  3 сэмпла, 2 признака, 2 класса
#
#  sign_0:  10, 20, 30   → min=10, max=30
#  sign_1: 100, 200, 300  → min=100, max=300
#
#  Формула: norm(x) = (x - xmin) / (xmax - xmin)
#
#  Нормализованные признаки:
#    Sample 0: [(10-10)/20, (100-100)/200] = [0.0, 0.0]
#    Sample 1: [(20-10)/20, (200-100)/200] = [0.5, 0.5]
#    Sample 2: [(30-10)/20, (300-100)/200] = [1.0, 1.0]
#
#  class_marks остаются без изменений.

SAMPLES = [
    Sample(signs=[10.0, 100.0], class_marks=[1.0, 0.0]),
    Sample(signs=[20.0, 200.0], class_marks=[0.0, 1.0]),
    Sample(signs=[30.0, 300.0], class_marks=[1.0, 0.0]),
]
SIGNS_COUNT = 2
CLASSES_COUNT = 2

EXPECTED_MINS = [10.0, 100.0]
EXPECTED_MAXS = [30.0, 300.0]

EXPECTED_NORMALIZED_SAMPLES = [
    Sample(signs=[0.0, 0.0], class_marks=[1.0, 0.0]),
    Sample(signs=[0.5, 0.5], class_marks=[0.0, 1.0]),
    Sample(signs=[1.0, 1.0], class_marks=[1.0, 0.0]),
]


def test_min_max_function():
    """
    min_max_function(x, xmin, xmax) = (x - xmin) / (xmax - xmin)

    (20 - 10) / (30 - 10) = 0.5
    (10 - 10) / (30 - 10) = 0.0
    (30 - 10) / (30 - 10) = 1.0
    (15 - 10) / (30 - 10) = 0.25
    """
    cases = [
        (20.0, 10.0, 30.0, 0.5),
        (10.0, 10.0, 30.0, 0.0),
        (30.0, 10.0, 30.0, 1.0),
        (15.0, 10.0, 30.0, 0.25),
    ]

    errors: List[Any] = list()
    for i, (x, xmin, xmax, expected) in enumerate(cases):
        result = min_max_function(x, xmin, xmax)
        if abs(result - expected) > TOLERANCE:
            logger.error(f" Test error. min_max_function case {i}: x={x}, xmin={xmin}, xmax={xmax}")
            errors.append({"case": i, "expected": expected, "received": result})

    if len(errors):
        raise TestException(f" errors: {errors}")
    else:
        logger.info("test_min_max_function complete!")


def test_min_max_signs_normalize():
    """
    signs=[20.0, 200.0], mins=[10.0, 100.0], maxs=[30.0, 300.0]

    norm_0 = (20 - 10) / (30 - 10)  = 0.5
    norm_1 = (200 - 100) / (300 - 100) = 0.5

    expected = [0.5, 0.5]
    """
    signs = [20.0, 200.0]
    expected = [0.5, 0.5]

    result = min_max_signs_normalize(
        signs=signs, maxs=EXPECTED_MAXS, mins=EXPECTED_MINS, signs_count=SIGNS_COUNT,
    )

    errors: List[Any] = list()
    for j in range(len(expected)):
        if abs(result[j] - expected[j]) > TOLERANCE:
            logger.error(f" Test error. min_max_signs_normalize index {j}")
            errors.append({"index": j, "expected": expected[j], "received": result[j]})

    if len(errors):
        raise TestException(f" errors: {errors}")
    else:
        logger.info("test_min_max_signs_normalize complete!")


def test_min_max_samples_normalize():
    """
    Проверяет min_max_samples_normalaize:
      1) mins и maxs вычислены верно
      2) все сэмплы нормализованы корректно
      3) class_marks не изменены
    """
    normed, mins, maxs = min_max_samples_normalaize(
        data=SAMPLES, signs_count=SIGNS_COUNT, classes_count=CLASSES_COUNT,
    )

    errors: List[Any] = list()

    for j in range(SIGNS_COUNT):
        if abs(mins[j] - EXPECTED_MINS[j]) > TOLERANCE:
            logger.error(f" Test error. mins[{j}]")
            errors.append({"param": f"mins[{j}]", "expected": EXPECTED_MINS[j], "received": mins[j]})
        if abs(maxs[j] - EXPECTED_MAXS[j]) > TOLERANCE:
            logger.error(f" Test error. maxs[{j}]")
            errors.append({"param": f"maxs[{j}]", "expected": EXPECTED_MAXS[j], "received": maxs[j]})

    for i in range(len(EXPECTED_NORMALIZED_SAMPLES)):
        exp_sample = EXPECTED_NORMALIZED_SAMPLES[i]

        for j in range(SIGNS_COUNT):
            if abs(normed[i].signs[j] - exp_sample.signs[j]) > TOLERANCE:
                logger.error(f" Test error. sample {i} sign {j}")
                errors.append({
                    "sample": i, "sign": j,
                    "expected": exp_sample.signs[j], "received": normed[i].signs[j],
                })

        for j in range(CLASSES_COUNT):
            if abs(normed[i].class_marks[j] - exp_sample.class_marks[j]) > TOLERANCE:
                logger.error(f" Test error. sample {i} class_mark {j}")
                errors.append({
                    "sample": i, "class_mark": j,
                    "expected": exp_sample.class_marks[j], "received": normed[i].class_marks[j],
                })

    if len(errors):
        raise TestException(f" errors: {errors}")
    else:
        logger.info("test_min_max_samples_normalize complete!")
