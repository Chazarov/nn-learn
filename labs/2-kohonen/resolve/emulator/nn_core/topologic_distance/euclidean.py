import numpy.typing as npt
import numpy as np

from nn_core.topologic_distance import ITopologicCalculator


class EuclideanTopologicDistance(ITopologicCalculator):
    def perform(self, winner_idx: int, neurons_count: int) -> npt.NDArray[np.float64]:
        indices = np.arange(neurons_count, dtype=np.float64) 
        return (winner_idx - indices) ** 2