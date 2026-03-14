import numpy.typing as npt
import numpy as np

from nn_core.topologic_distance import ITopologicCalculator

class ManhattanTopologicDistance(ITopologicCalculator):
    def perform(self, winner_idx: int, neurons_count: int) -> npt.NDArray[np.float64]:
        indices = np.arange(neurons_count, dtype=np.float64)  # ✅ float64 с самого начала
        result = np.abs(winner_idx - indices)                 # abs сохраняет float64
        return result