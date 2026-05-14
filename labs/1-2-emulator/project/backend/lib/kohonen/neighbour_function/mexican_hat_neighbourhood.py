from lib.kohonen.neighbour_function.base import INeighbourFunction
from lib.kohonen.models.enums import NeighbourhoodFunctionType
import numpy.typing as npt
import numpy as np

class MexicanHatNeighborhood(INeighbourFunction):
    def perform(self, topo_dist: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
        excitation = 1 - topo_dist**2 / sigma**2
        gaussian = np.exp(-topo_dist**2 / (2 * sigma**2))
        result = excitation * gaussian
        result[topo_dist >= sigma] = 0
        return result
    def get_type(self)-> str: return NeighbourhoodFunctionType.MEXICAN_HAT

