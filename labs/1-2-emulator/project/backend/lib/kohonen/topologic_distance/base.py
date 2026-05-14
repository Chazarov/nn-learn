from abc import ABC, abstractmethod
from typing import Tuple

import numpy.typing as npt
import numpy as np



class ITopologicCalculator(ABC):
    """Топологическое расстояние на 2D-решётке m×n.

    Нейроны индексируются линейно: ``idx = row * cols + col``.
    Метод :meth:`perform` возвращает 1D-массив длины ``neurons_count``
    (= ``rows * cols``) с расстояниями от нейрона-победителя до всех остальных.
    """

    def __init__(self, cols: int) -> None:
        if cols <= 0:
            raise ValueError(f"cols must be > 0, got {cols}")
        self.cols = cols

    def _grid_coords(
        self, neurons_count: int
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        if neurons_count <= 0 or neurons_count % self.cols != 0:
            raise ValueError(
                "neurons_count must be a positive multiple of cols "
                f"(neurons_count={neurons_count}, cols={self.cols})"
            )
        idx = np.arange(neurons_count, dtype=np.int64)
        return idx // self.cols, idx % self.cols

    @abstractmethod
    def perform(
        self, winner_idx: int, neurons_count: int
    ) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def get_type(self)-> str: pass
