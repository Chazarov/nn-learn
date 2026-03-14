from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np


class INeighbourFunction(ABC):

    @abstractmethod
    def perform(self, topo_dist: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:pass 

class GaussianNEighborhood(INeighbourFunction):
    def perform(self, topo_dist: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
        return np.exp(-topo_dist**2 / (2 * sigma**2))
    
class MexicanHatNeighborhood(INeighbourFunction):
    def perform(self, topo_dist: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
        excitation = 1 - topo_dist**2 / sigma**2
        gaussian = np.exp(-topo_dist**2 / (2 * sigma**2))
        result = excitation * gaussian
        result[topo_dist >= sigma] = 0
        return result