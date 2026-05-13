import numpy as np
import numpy.typing as npt

from exceptions import ArgumentException
from log import logger


def min_max_normalize(
    input_vector: npt.NDArray[np.float64],
    mins: npt.NDArray[np.float64],
    maxs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Покомпонентная min-max нормализация вектора.

    **Формула.** Для каждого индекса признака ``j`` (``j = 0 … d - 1``):

    ::

        y[j] = (x[j] - mins[j]) / (maxs[j] - mins[j])   при maxs[j] != mins[j],
        y[j] = 0                                         при maxs[j] == mins[j].

    **Где:**

    * ``j`` — индекс признака;
    * ``d`` — число признаков: одинаковая длина ``input_vector``, ``mins`` и ``maxs``
      после приведения к одномерному виду;
    * ``x[j]`` — ``j``-я компонента входа (значение из ``input_vector``);
    * ``mins[j]`` — нижняя граница шкалы для признака ``j``;
    * ``maxs[j]`` — верхняя граница шкалы для признака ``j``;
    * ``y[j]`` — ``j``-я компонента возвращаемого вектора.

    Если ``maxs[j] != mins[j]``, то ``y[j]`` — линейное отображение ``x[j]`` с отрезка
    ``[mins[j], maxs[j]]`` на числовую прямую; при ``x[j]`` вне этого отрезка ``y[j]``
    может быть меньше 0 или больше 1.
    """
    x = np.asarray(input_vector, dtype=np.float64).ravel()
    lo = np.asarray(mins, dtype=np.float64).ravel()
    hi = np.asarray(maxs, dtype=np.float64).ravel()
    d = x.size
    if lo.size != d or hi.size != d:
        e_str = (
            f"Длины должны совпадать: len(input_vector)={d}, len(mins)={lo.size}, len(maxs)={hi.size}"
        )
        logger.error(e_str)
        raise ArgumentException(e_str)

    denom = hi - lo
    y = np.zeros(d, dtype=np.float64)
    nz = denom != 0.0
    y[nz] = (x[nz] - lo[nz]) / denom[nz]
    return y
