import numpy.typing as npt
import numpy as np



from abc import ABC, abstractmethod


class IVectorDistanceCalculator(ABC):

    @abstractmethod
    def perform(self, weights: npt.NDArray[np.float64],
                 input_vector:npt.NDArray[np.float64]) -> npt.NDArray[np.float64] : pass

