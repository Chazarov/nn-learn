import numpy as np
import numpy.typing as npt

from lib.kohonen.neighbour_function.base import INeighbourFunction
from lib.kohonen.models.enums import NeighbourhoodFunctionType




class GaussianNEighborhood(INeighbourFunction):
    def perform(self, topo_dist: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
        return np.exp(-topo_dist**2 / (2 * sigma**2))
    def get_type(self)-> str: return NeighbourhoodFunctionType.GAUSSIAN
