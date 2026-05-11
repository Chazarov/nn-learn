from typing import List

import cv2
import numpy as np
import numpy.typing as npt


def get_component_planes_visualisation(
    weights: npt.NDArray[np.float64],
    input_names: List[str],
    input_id: int,
    samples: npt.NDArray[np.float64],
) -> npt.NDArray[np.uint8]:
    num_neurons = weights.shape[0]
    grid_rows = int(np.sqrt(num_neurons))
    grid_cols = (num_neurons + grid_rows - 1) // grid_rows
    values = weights[:, input_id].copy()
    if grid_rows * grid_cols > num_neurons:
        pad = grid_rows * grid_cols - num_neurons
        values = np.pad(values, (0, pad), mode="edge")
    grid = values.reshape(grid_rows, grid_cols)
    vmin, vmax = grid.min(), grid.max()
    if vmax - vmin > 1e-10:
        grid = ((grid - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        grid = np.full_like(grid, 128, dtype=np.uint8)
    cell_size = 32
    img = cv2.resize(
        grid, (grid_cols * cell_size, grid_rows * cell_size), interpolation=cv2.INTER_NEAREST
    )
    return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)


def get_u_matrix_visualisation(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    num_neurons = weights.shape[0]
    grid_rows = int(np.sqrt(num_neurons))
    grid_cols = (num_neurons + grid_rows - 1) // grid_rows
    indices = np.arange(num_neurons).reshape(grid_rows, grid_cols)
    u_matrix = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for i in range(grid_rows):
        for j in range(grid_cols):
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_rows and 0 <= nj < grid_cols:
                    neighbors.append(np.linalg.norm(weights[indices[i, j]] - weights[indices[ni, nj]]))
            u_matrix[i, j] = np.mean(neighbors) if neighbors else 0.0
    vmin, vmax = u_matrix.min(), u_matrix.max()
    if vmax - vmin > 1e-10:
        u_norm = ((u_matrix - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        u_norm = np.full_like(u_matrix, 128, dtype=np.uint8)
    r = 24
    h = np.sqrt(3) * r
    w = 1.5 * r
    img_w = int(grid_cols * 2 * w + r)
    img_h = int(grid_rows * h + r)
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:] = 255
    for i in range(grid_rows):
        for j in range(grid_cols):
            cx = j * 2 * w + (r if i % 2 == 1 else 0)
            cy = i * h + r
            angles = np.linspace(0, 2 * np.pi, 7)[:-1]
            pts = np.column_stack([
                cx + r * np.cos(angles),
                cy + r * np.sin(angles),
            ]).astype(np.int32)
            color = tuple(int(x) for x in cv2.applyColorMap(np.array([[u_norm[i, j]]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0].tolist())
            cv2.fillPoly(img, [pts], color)
            cv2.polylines(img, [pts], True, (80, 80, 80), 1)
    return img
    