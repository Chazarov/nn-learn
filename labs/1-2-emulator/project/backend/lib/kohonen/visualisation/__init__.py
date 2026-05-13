from typing import List

import cv2
import numpy as np
import numpy.typing as npt


def _validate_grid(weights: npt.NDArray[np.float64], rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows/cols must be > 0, got rows={rows}, cols={cols}")
    if weights.shape[0] != rows * cols:
        raise ValueError(
            "weights.shape[0] must equal rows*cols "
            f"(got {weights.shape[0]} vs {rows}*{cols}={rows * cols})"
        )


def get_component_planes_visualisation(
    weights: npt.NDArray[np.float64],
    input_names: List[str],
    input_id: int,
    samples: npt.NDArray[np.float64],
    rows: int,
    cols: int,
    cell_size: int = 32,
) -> npt.NDArray[np.uint8]:
    """Раскраска карты Кохонена по одному входному признаку (component plane).

    Сетка задаётся явно ``rows × cols``; матрица весов должна иметь
    ``rows * cols`` строк.
    """
    _validate_grid(weights, rows, cols)
    grid = weights[:, input_id].reshape(rows, cols)
    vmin, vmax = float(grid.min()), float(grid.max())
    if vmax - vmin > 1e-10:
        grid = ((grid - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        grid = np.full_like(grid, 128, dtype=np.uint8)
    img = cv2.resize(
        grid,
        (cols * cell_size, rows * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)


def get_u_matrix_visualisation(
    weights: npt.NDArray[np.float64],
    rows: int,
    cols: int,
) -> npt.NDArray[np.uint8]:
    """U-matrix визуализация для прямоугольной сетки ``rows × cols``.

    Соседство 4-связное (без диагоналей); расстояние — евклидово в пространстве
    весов. Сетка отрисовывается шестиугольниками с горизонтальным сдвигом
    нечётных строк (классический «sombrero»-стиль).
    """
    _validate_grid(weights, rows, cols)
    indices = np.arange(rows * cols).reshape(rows, cols)
    u_matrix = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append(
                        np.linalg.norm(
                            weights[indices[i, j]] - weights[indices[ni, nj]]
                        )
                    )
            u_matrix[i, j] = float(np.mean(neighbors)) if neighbors else 0.0
    vmin, vmax = float(u_matrix.min()), float(u_matrix.max())
    if vmax - vmin > 1e-10:
        u_norm = ((u_matrix - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        u_norm = np.full_like(u_matrix, 128, dtype=np.uint8)
    r = 24
    h = np.sqrt(3) * r
    w = 1.5 * r
    img_w = int(cols * 2 * w + r)
    img_h = int(rows * h + r)
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:] = 255
    for i in range(rows):
        for j in range(cols):
            cx = j * 2 * w + (r if i % 2 == 1 else 0)
            cy = i * h + r
            angles = np.linspace(0, 2 * np.pi, 7)[:-1]
            pts = np.column_stack(
                [
                    cx + r * np.cos(angles),
                    cy + r * np.sin(angles),
                ]
            ).astype(np.int32)
            color = tuple(
                int(x)
                for x in cv2.applyColorMap(
                    np.array([[u_norm[i, j]]], dtype=np.uint8),
                    cv2.COLORMAP_VIRIDIS,
                )[0, 0].tolist()
            )
            cv2.fillPoly(img, [pts], color)
            cv2.polylines(img, [pts], True, (80, 80, 80), 1)
    return img
