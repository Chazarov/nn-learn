from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class INeighbourFunction(ABC):

    @abstractmethod
    def perform(self, topo_dist: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:pass 

    
    @abstractmethod
    def get_type(self)-> str: pass
