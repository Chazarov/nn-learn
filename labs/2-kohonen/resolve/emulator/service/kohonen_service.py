import numpy as np
import numpy.typing as npt

from nn_core.topologic_distance import ITopologicCalculator
from nn_core.neighbour_function import INeighbourFunction
from nn_core.vector_distance_calculation import IVectorDistanceCalculator

class KohonenNetworkService:

    def init_network(
        self,
        clusters_count:int,
        output_size:int,
        input_size:int) -> npt.NDArray[np.float64]:
        pass

    def train( 
        self,
        weights: npt.NDArray[np.float64],
        samples: npt.NDArray[np.float64], # список входных векторов для обучения
        learning_rate: float,
        vector_distance_calc: IVectorDistanceCalculator,
        top_dist_calc: ITopologicCalculator,
        neighbour_func: INeighbourFunction
    )  -> npt.NDArray[np.float64]:
       
        pass

    def predict(
        self,
        weights: npt.NDArray[np.float64],
        input_vector: npt.NDArray[np.float64],
        vector_distance_calc: IVectorDistanceCalculator,
        mins: npt.NDArray[np.float64],
        maxs: npt.NDArray[np.float64],
        input_size: int,
    ) -> npt.NDArray[np.float64]:
        pass


    def get_component_matrix_visualisation(
        self,
        weights: npt.NDArray[np.float64],
        input_id:int,# номер признака который необходимо визуализировать
        samples: npt.NDArray[np.float64], # список входных векторов для обучения
    ) -> npt.NDArray[np.uint8]:
        pass
        
    def get_u_matrix_visualisation(
        self,
        weights: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        pass