import numpy as np
import numpy.typing as npt

from lib.kohonen.topologic_distance.base import ITopologicCalculator
from lib.kohonen.models.enums import TopologyDistanceType


class EuclideanTopologicDistance(ITopologicCalculator):
    """Евклидово топологическое расстояние на 2D-решётке m×n.

    Для линейного индекса ``i`` координаты на сетке вычисляются как
    ``(r, c) = divmod(i, cols)``. Расстояние возвращается без возведения
    в квадрат: квадрат уже сидит внутри гауссовой функции соседства.
    """

    def perform(
        self, winner_idx: int, neurons_count: int
    ) -> npt.NDArray[np.float64]:
        wr, wc = divmod(winner_idx, self.cols)
        r, c = self._grid_coords(neurons_count)
        return np.sqrt((r - wr) ** 2 + (c - wc) ** 2).astype(np.float64)

    def get_type(self)-> str: return TopologyDistanceType.EUCLIDEAN
