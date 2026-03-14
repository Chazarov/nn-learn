import numpy as np
import numpy.typing as npt

from nn_core.topologic_distance import ITopologicCalculator
from nn_core.neighbour_function import INeighbourFunction
from nn_core.vector_distance_calculation import IVectorDistanceCalculator
from nn_core.visualisation import get_component_planes_visualisation, get_u_matrix_visualisation
from nn_core.weights_updator import WeightApdator


class KohonenNetworkService:

    def init_network(
        self,
        clusters_count: int,
        output_size: int,
        input_size: int,
    ) -> npt.NDArray[np.float64]:
        return np.random.rand(clusters_count, input_size).astype(np.float64)

    def train(
        self,
        weights: npt.NDArray[np.float64],
        samples: npt.NDArray[np.float64],
        learning_rate: float,
        vector_distance_calc: IVectorDistanceCalculator,
        top_dist_calc: ITopologicCalculator,
        neighbour_func: INeighbourFunction,
    ) -> npt.NDArray[np.float64]:
        weights = weights.copy()
        weight_updator = WeightApdator(top_dist_calc, neighbour_func, learning_rate)
        for sample in samples:
            distances = vector_distance_calc.perform(weights, sample)
            output_vector = -distances
            weight_updator.update_weights(weights, learning_rate, output_vector, sample)
        return weights

    def predict(
        self,
        weights: npt.NDArray[np.float64],
        input_vector: npt.NDArray[np.float64],
        vector_distance_calc: IVectorDistanceCalculator,
        mins: npt.NDArray[np.float64],
        maxs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        normalized = (input_vector - mins) / (maxs - mins + 1e-10)
        distances = vector_distance_calc.perform(weights, normalized)
        winner_idx = np.argmin(distances)
        return np.array([float(winner_idx)], dtype=np.float64)

    def get_component_matrix_visualisation(
        self,
        weights: npt.NDArray[np.float64],
        input_id: int,
        samples: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        return get_component_planes_visualisation(
            weights, [], input_id, samples
        )

    def get_u_matrix_visualisation(
        self,
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        return get_u_matrix_visualisation(weights)