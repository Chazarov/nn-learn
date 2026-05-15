from typing import List, cast

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
    return cast(
        npt.NDArray[np.uint8],
        cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS),
    )


def _u_matrix_neighbor_mean(
    weights: npt.NDArray[np.float64], rows: int, cols: int
) -> npt.NDArray[np.float64]:
    """Средняя длина разности весов с соседями по 4-связности (как у карты ``rows × cols``)."""
    _validate_grid(weights, rows, cols)
    indices = np.arange(rows * cols).reshape(rows, cols)
    u_matrix = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            neighbors: List[float] = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append(
                        float(
                            np.linalg.norm(
                                weights[indices[i, j]] - weights[indices[ni, nj]]
                            )
                        )
                    )
            u_matrix[i, j] = float(np.mean(neighbors)) if neighbors else 0.0
    return u_matrix


def _normalize_gray_u8(u_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    vmin, vmax = float(u_matrix.min()), float(u_matrix.max())
    if vmax - vmin > 1e-10:
        return ((u_matrix - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return np.full_like(u_matrix, 128, dtype=np.uint8)


def get_u_matrix_visualisation(
    weights: npt.NDArray[np.float64],
    rows: int,
    cols: int,
    cell_size: int = 32,
    grid_line: bool = True,
) -> npt.NDArray[np.uint8]:
    """U-matrix: квадратная сетка ``rows × cols`` (как component planes).

    Используется в приложении по умолчанию.
    """
    u_matrix = _u_matrix_neighbor_mean(weights, rows, cols)
    u_norm = _normalize_gray_u8(u_matrix)
    img = cv2.resize(
        u_norm,
        (cols * cell_size, rows * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    bgr = cast(npt.NDArray[np.uint8], cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS))
    if grid_line and cell_size >= 4:
        h, w = bgr.shape[:2]
        line_color = (72, 72, 72)
        for i in range(rows + 1):
            y = min(i * cell_size, h - 1)
            cv2.line(bgr, (0, y), (w - 1, y), line_color, 1, lineType=cv2.LINE_AA)
        for j in range(cols + 1):
            x = min(j * cell_size, w - 1)
            cv2.line(bgr, (x, 0), (x, h - 1), line_color, 1, lineType=cv2.LINE_AA)
    return bgr


def _hex_vertices_pointy_top(cx: float, cy: float, radius: float) -> npt.NDArray[np.int32]:
    """Вершины правильного шестиугольника (pointy-top), ``radius`` — от центра до вершины."""
    k = np.arange(6, dtype=np.float64)
    # Первая вершина «вверх» по экрану (уменьшение y в системе координат изображения).
    angles = -0.5 * np.pi + k * (np.pi / 3.0)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.column_stack((xs, ys)).astype(np.int32)


def get_u_matrix_visualisation_hex(
    weights: npt.NDArray[np.float64],
    rows: int,
    cols: int,
    hex_radius: float = 22.0,
    pad: int = 6,
) -> npt.NDArray[np.uint8]:
    """U-matrix с гексагональной отрисовкой (альтернативный вид; в приложении не используется).

    Топология расчёта — та же прямоугольная сетка ``rows × cols`` (4-соседство);
    только способ рисования — плотная гексагональная упаковка (pointy-top).

    См. https://www.redblobgames.com/grids/hexagons/#spacing
    """
    u_matrix = _u_matrix_neighbor_mean(weights, rows, cols)
    u_norm = _normalize_gray_u8(u_matrix)

    r = float(hex_radius)
    sqrt3 = float(np.sqrt(3.0))
    dx = sqrt3 * r
    dy = 1.5 * r
    odd_row_shift = 0.5 * sqrt3 * r  # == dx / 2

    half_w = 0.5 * sqrt3 * r
    centers: list[tuple[float, float]] = []
    for i in range(rows):
        for j in range(cols):
            cx = float(pad + half_w + j * dx + (i % 2) * odd_row_shift)
            cy = float(pad + r + i * dy)
            centers.append((cx, cy))

    max_cx = max(c[0] for c in centers) if centers else float(pad)
    max_cy = max(c[1] for c in centers) if centers else float(pad)
    img_w = int(np.ceil(max_cx + half_w + pad))
    img_h = int(np.ceil(max_cy + r + pad))

    img = np.full((img_h, img_w, 3), 0, dtype=np.uint8)

    for (cx, cy), val in zip(centers, u_norm.ravel(order="C"), strict=True):
        pts = _hex_vertices_pointy_top(cx, cy, r)
        cmap_bgr = cast(
            npt.NDArray[np.uint8],
            cv2.applyColorMap(
                np.array([[int(val)]], dtype=np.uint8),
                cv2.COLORMAP_VIRIDIS,
            ),
        )
        color = (int(cmap_bgr[0, 0, 0]), int(cmap_bgr[0, 0, 1]), int(cmap_bgr[0, 0, 2]))
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (72, 72, 72), 1, lineType=cv2.LINE_AA)

    return img
