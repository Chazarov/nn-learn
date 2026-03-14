

from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np

class ITopologicCalculator(ABC):
    @abstractmethod
    def perform(self, winner_idx:int, neurons_count:int) -> npt.NDArray[np.float64]: pass

