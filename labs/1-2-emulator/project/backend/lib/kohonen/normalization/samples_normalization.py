"""Min-max нормализация матрицы примеров (строки — образы, столбцы — признаки)."""

from typing import Tuple

import numpy as np
import numpy.typing as npt

from exceptions import ArgumentException
from log import logger


def min_max_bounds_from_samples(
    samples: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    По каждому признаку — минимум и максимум по всей выборке (для последующей min-max
    нормализации с теми же границами при обучении и инференсе).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        e_str = f"samples must be 2D (n_samples, n_features), got shape {x.shape}"
        logger.error(e_str)
        raise ArgumentException(e_str)
    if x.shape[0] == 0:
        e_str = "samples must contain at least one row"
        logger.error(e_str)
        raise ArgumentException(e_str)
    return x.min(axis=0).astype(np.float64), x.max(axis=0).astype(np.float64)


def normalize_samples_min_max(
    samples: npt.NDArray[np.float64],
    mins: npt.NDArray[np.float64],
    maxs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Построчно та же min-max нормализация, что :func:`min_max_normalize` в
    ``weights_normalization``: для строки ``i`` и признака ``j`` —
    ``(X[i,j] - mins[j]) / (maxs[j] - mins[j])`` при ``maxs[j] != mins[j]``, иначе ``0``.

    Возвращает новый массив ``(n_samples, n_features)``, вход ``samples`` не изменяется.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        e_str = f"samples must be 2D (n_samples, n_features), got shape {x.shape}"
        logger.error(e_str)
        raise ArgumentException(e_str)
    d = x.shape[1]
    lo = np.asarray(mins, dtype=np.float64).ravel()
    hi = np.asarray(maxs, dtype=np.float64).ravel()
    if lo.size != d or hi.size != d:
        e_str = (
            f"mins/maxs length must match n_features={d}, "
            f"got len(mins)={lo.size}, len(maxs)={hi.size}"
        )
        logger.error(e_str)
        raise ArgumentException(e_str)

    denom = hi - lo
    out = np.empty_like(x, dtype=np.float64)
    nz = denom != 0.0
    out[:, nz] = (x[:, nz] - lo[nz]) / denom[nz]
    out[:, ~nz] = 0.0
    return out
