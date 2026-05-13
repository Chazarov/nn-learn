"""Линейное расписание скорости обучения для SOM."""

from exceptions import ArgumentException
from log import logger


def decreasing_linear_rate(
    step_index: int,
    total_steps: int,
    rate_start: float,
    rate_end: float,
) -> float:
    """
    Линейная интерполяция скорости обучения по индексу шага (см. readme.md в этом пакете).

    При ``total_steps >= 2`` и ``rate_start > rate_end`` получается линейно убывающая
    скорость от ``rate_start`` к ``rate_end``.
    """
    if total_steps < 1:
        e_str = f"total_steps must be >= 1, got {total_steps}"
        logger.error(e_str)
        raise ArgumentException(e_str)
    if not (0.0 < rate_start <= 1.0 and 0.0 < rate_end <= 1.0):
        e_str = "rate_start and rate_end must satisfy 0 < rate <= 1"
        logger.error(e_str)
        raise ArgumentException(e_str)
    if rate_start <= rate_end:
        e_str = (
            f"rate_start must be > rate_end for decreasing schedule, got {rate_start} <= {rate_end}"
        )
        logger.error(e_str)
        raise ArgumentException(e_str)

    k = min(max(int(step_index), 0), total_steps - 1)
    if total_steps == 1:
        return float(rate_start)

    t = k / float(total_steps - 1)
    return float(rate_start + (rate_end - rate_start) * t)
