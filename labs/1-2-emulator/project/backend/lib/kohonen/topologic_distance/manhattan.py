import numpy as np
import numpy.typing as npt

from lib.kohonen.topologic_distance.base import ITopologicCalculator
from lib.kohonen.models.enums import TopologyDistanceType

class ManhattanTopologicDistance(ITopologicCalculator):
    """Манхэттенское топологическое расстояние на 2D-решётке m*n."""

    def perform(
        self, winner_idx: int, neurons_count: int
    ) -> npt.NDArray[np.float64]:
        wr, wc = divmod(winner_idx, self.cols)
        r, c = self._grid_coords(neurons_count)
        return (np.abs(r - wr) + np.abs(c - wc)).astype(np.float64)

    def get_type(self)-> str: return TopologyDistanceType.MANHATTAN