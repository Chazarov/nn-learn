from enum import Enum
import numpy as np
import cv2
import numpy.typing as npt
from typing import Generator, List, Optional, Tuple

CELL: int = 50
MARGIN: int = 80
NEURON_GAP: int = 60
NEURON_R: int = 15
WEIGHT_R: int = 14

_COLORMAP: List[Tuple[float, Tuple[int, int, int]]] = [
    (-10.0, (30, 0, 128)),      # темно-бордовый
    (-7.5,  (50, 0, 100)),      # бордовый
    (-5.0,  (0, 0, 255)),       # синий
    (-3.75, (100, 50, 200)),    # фиолетовый
    (-3.0,  (150, 0, 200)),     # пурпурный
    (-2.5,  (255, 50, 50)),     # красный
    (-2.0,  (255, 100, 100)),   # светло-красный
    (-1.5,  (255, 150, 150)),   # розовый
    (-1.0,  (255, 200, 200)),   # очень светло-розовый
    (-0.8,  (255, 220, 220)),   # почти белый розовый
    (-0.6,  (255, 235, 240)),   # бледно-розовый
    (-0.4,  (255, 245, 255)),   # почти белый
    (-0.2,  (245, 255, 255)),   # голубовато-белый
    (0.0,   (255, 255, 255)),   # чистый белый
    (0.2,   (240, 255, 255)),   # светло-голубой
    (0.4,   (220, 245, 255)),   # голубой
    (0.5,   (200, 235, 255)),   # ярко-голубой
    (0.6,   (150, 220, 255)),   # небесно-голубой ← отличается от 0.5
    (0.7,   (100, 200, 255)),   # лазурный
    (0.8,   (50, 180, 255)),    # бирюзовый ← отличается от 0.6
    (0.9,   (0, 160, 255)),     # яркий бирюзовый
    (1.0,   (0, 140, 240)),     # сине-бирюзовый
    (1.5,   (0, 180, 200)),     # бирюза
    (2.0,   (0, 165, 255)),     # оранжево-голубой
    (2.5,   (100, 200, 150)),   # светло-бирюзово-зеленый
    (3.0,   (144, 238, 144)),   # светло-зеленый
    (4.0,   (100, 220, 100)),   # ярко-зеленый
    (5.0,   (0, 200, 0)),       # зеленый
    (7.5,   (0, 150, 50)),      # темно-зеленый
    (10.0,  (0, 100, 0)),       # очень темно-зеленый
]


def _weight_to_bgr(value: float) -> Tuple[int, int, int]:
    """Diverging colormap: weight in [-10, 10] -> BGR color."""
    value = max(-10.0, min(10.0, value))
    for idx in range(len(_COLORMAP) - 1):
        v0, c0 = _COLORMAP[idx]
        v1, c1 = _COLORMAP[idx + 1]
        if value <= v1:
            t = (value - v0) / (v1 - v0) if v1 != v0 else 0.0
            return (
                int(c0[0] + (c1[0] - c0[0]) * t),
                int(c0[1] + (c1[1] - c0[1]) * t),
                int(c0[2] + (c1[2] - c0[2]) * t),
            )
    return _COLORMAP[-1][1]


def _grid_dims(perceptron: List[List[List[float]]], q: int) -> Tuple[int, int]:
    """Visual grid (rows, cols) for layer q depending on snake direction."""
    nt = len(perceptron[q])
    ns = len(perceptron[q][0])
    if q % 4 in (0, 2):  # DOWN / UP — transposed
        return ns, nt
    return nt, ns  # RIGHT — direct

class ColorTheme(str, Enum):
    DARK = "dark"
    WHITE = "white"
    


def get_visualisation(perceptron: List[List[List[float]]], color_theme:ColorTheme = ColorTheme.WHITE) -> npt.NDArray[np.uint8]:
    num_layers: int = len(perceptron)
    if num_layers == 0:
        return np.full((100, 100, 3), 245, dtype=np.uint8)

    neuron_counts: List[int] = [len(perceptron[0][0])]
    for q in range(num_layers):
        neuron_counts.append(len(perceptron[q]))

    def is_top(q: int) -> bool:
        return q % 4 in (0, 3)

    def _top_rows() -> Generator[int, None, None]:
        return (_grid_dims(perceptron, q)[0] for q in range(num_layers) if is_top(q))

    def _bot_rows() -> Generator[int, None, None]:
        return (_grid_dims(perceptron, q)[0] for q in range(num_layers) if not is_top(q))

    max_top: int = max(_top_rows(), default=0)
    max_bot: int = max(_bot_rows(), default=0)

    mid_y: int = MARGIN + (max_top - 1) * CELL + NEURON_GAP
    canvas_h: float = mid_y + MARGIN
    if max_bot > 0:
        canvas_h = mid_y + NEURON_GAP + (max_bot - 1) * CELL + MARGIN

    def y_top(i: int, n: int) -> int:
        return mid_y - NEURON_GAP - (n - 1 - i) * CELL

    def y_bot(i: int) -> int:
        return mid_y + NEURON_GAP + i * CELL

    weight_pts: List[Tuple[int, int, float]] = []
    neuron_pts: List[Tuple[int, int]] = []

    x: int = MARGIN

    for pair_start in range(0, num_layers, 2):
        qv: int = pair_start
        qh: Optional[int] = pair_start + 1 if pair_start + 1 < num_layers else None

        rows_v, cols_v = _grid_dims(perceptron, qv)
        n_src: int = neuron_counts[qv]

        # --- source neurons (vertical column) ---
        for i in range(n_src):
            ny: int = y_top(i, n_src) if qv % 4 == 0 else y_bot(i)
            neuron_pts.append((x, ny))

        x += NEURON_GAP
        gx: int = x

        # --- vertical weight grid (DOWN / UP) ---
        for r in range(rows_v):
            for c in range(cols_v):
                wx: int = gx + c * CELL
                wy: int = y_top(r, rows_v) if is_top(qv) else y_bot(r)
                weight_pts.append((wx, wy, perceptron[qv][c][r]))

        # --- middle-line neurons ---
        n_mid: int = neuron_counts[qv + 1]
        for j in range(n_mid):
            neuron_pts.append((gx + j * CELL, mid_y))

        # --- horizontal weight grid (RIGHT) ---
        if qh is not None:
            rows_h, cols_h = _grid_dims(perceptron, qh)

            for r in range(rows_h):
                for c in range(cols_h):
                    wx = gx + c * CELL
                    wy = y_top(r, rows_h) if is_top(qh) else y_bot(r)
                    weight_pts.append((wx, wy, perceptron[qh][r][c]))

            # target neurons of RIGHT grid (vertical column to the right)
            n_tgt: int = neuron_counts[qh + 1]
            tx: int = gx + (cols_h - 1) * CELL + NEURON_GAP

            for i in range(n_tgt):
                ny = y_top(i, rows_h) if is_top(qh) else y_bot(i)
                neuron_pts.append((tx, ny))

            x = tx
        else:
            x = gx + (n_mid - 1) * CELL

    canvas_w: int = int(x + MARGIN)
    canvas_h_int: int = int(canvas_h)

    back_color = None
    if color_theme == ColorTheme.DARK:
        back_color = 0
    elif color_theme == ColorTheme.WHITE:
        back_color = 245
    canvas: npt.NDArray[np.uint8] = np.full((canvas_h_int, canvas_w, 3), back_color, dtype=np.uint8)

    for wx, wy, val in weight_pts:
        clr: Tuple[int, int, int] = _weight_to_bgr(val)
        cv2.circle(canvas, (wx, wy), WEIGHT_R, clr, -1, cv2.LINE_AA)

    for nx, ny in neuron_pts:
        # cv2.circle(canvas, (nx, ny), NEURON_R, (255, 255, 255), -1, cv2.LINE_AA)
        # cv2.circle(canvas, (nx, ny), NEURON_R, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, 
              (nx - NEURON_R, ny - NEURON_R), 
              (nx + NEURON_R, ny + NEURON_R), 
              (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


