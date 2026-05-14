"""Линейная инициализация весов SOM по первым двум главным компонентам (в min-max шкале выборки)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from lib.kohonen.normalization import normalize_samples_min_max


def _grid_coord_1d(index: int, size: int) -> float:
    if size <= 1:
        return 0.0
    return -1.0 + 2.0 * float(index) / float(size - 1)


def _orthonormal_second(
    v1: npt.NDArray[np.float64], d: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    for _ in range(50):
        a = rng.standard_normal(d).astype(np.float64)
        a -= float(np.dot(a, v1)) * v1
        n = float(np.linalg.norm(a))
        if n > 1e-12:
            return (a / n).astype(np.float64)
    e = np.zeros(d, dtype=np.float64)
    e[int(np.argmax(np.abs(v1)))] = 1.0
    a = e - float(np.dot(e, v1)) * v1
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        e = np.eye(d, dtype=np.float64)[0]
        a = e - float(np.dot(e, v1)) * v1
        n = float(np.linalg.norm(a))
    return (a / n).astype(np.float64)


def initialize_som_weights_pca_grid(
    samples: npt.NDArray[np.float64],
    mins: npt.NDArray[np.float64],
    maxs: npt.NDArray[np.float64],
    rows: int,
    cols: int,
    rng: Optional[np.random.Generator] = None,
) -> npt.NDArray[np.float64]:
    """
    Веса ``(rows * cols, d)`` в том же min-max нормализованном пространстве, что и ``train``:
    PCA по ``normalize_samples_min_max(samples, mins, maxs)``, сетка ``[-1, 1]^2`` по PC1/PC2.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = normalize_samples_min_max(
        np.asarray(samples, dtype=np.float64),
        np.asarray(mins, dtype=np.float64).ravel(),
        np.asarray(maxs, dtype=np.float64).ravel(),
    )
    if z.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {z.shape}")
    n, d = z.shape
    if n < 1 or d < 1:
        raise ValueError(f"invalid shape {z.shape}")
    n_neurons = rows * cols
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows/cols must be > 0: rows={rows}, cols={cols}")

    if n == 1:
        return np.tile(z[0], (n_neurons, 1)).astype(np.float64)

    if d == 1:
        lo, hi = float(z.min()), float(z.max())
        if hi <= lo:
            w0 = np.full((n_neurons, 1), lo, dtype=np.float64)
        else:
            w0 = np.empty((n_neurons, 1), dtype=np.float64)
            for k in range(n_neurons):
                r, c = divmod(k, cols)
                t = 0.5 * (_grid_coord_1d(r, rows) + _grid_coord_1d(c, cols))
                w0[k, 0] = lo + (t + 1.0) * 0.5 * (hi - lo)
        return w0

    mean_z = z.mean(axis=0)
    zc = z - mean_z
    scale = float(np.max(np.abs(zc)))
    eps = max(1e-12, 1e-9 * max(scale, 1.0))

    if scale < eps:
        return np.tile(mean_z, (n_neurons, 1)).astype(np.float64)

    _, _, vh = np.linalg.svd(zc, full_matrices=False)
    v1 = vh[0].astype(np.float64)
    n1 = float(np.linalg.norm(v1))
    if n1 < 1e-12:
        v1 = np.zeros(d, dtype=np.float64)
        v1[0] = 1.0
    else:
        v1 /= n1

    if vh.shape[0] >= 2:
        v2 = vh[1].astype(np.float64)
        v2 -= float(np.dot(v2, v1)) * v1
        n2 = float(np.linalg.norm(v2))
        if n2 > 1e-12:
            v2 /= n2
        else:
            v2 = _orthonormal_second(v1, d, rng)
    else:
        v2 = _orthonormal_second(v1, d, rng)

    span1 = float(np.max(np.abs(zc @ v1)))
    span2 = float(np.max(np.abs(zc @ v2)))
    span1 = max(span1, eps)
    if span2 < eps:
        span2 = max(eps, 0.05 * span1)

    weights = np.empty((n_neurons, d), dtype=np.float64)
    for k in range(n_neurons):
        r, c = divmod(k, cols)
        g1 = _grid_coord_1d(r, rows)
        g2 = _grid_coord_1d(c, cols)
        weights[k] = mean_z + g1 * span1 * v1 + g2 * span2 * v2

    return weights
