import numpy as np
import numpy.typing as npt

from lib.kohonen.topologic_distance import ITopologicCalculator
from lib.kohonen.neighbour_function import INeighbourFunction
from lib.kohonen.vector_distance_calculation import IVectorDistanceCalculator
from lib.kohonen.visualisation import (
    get_component_planes_visualisation,
    get_u_matrix_visualisation,
)
from lib.kohonen.weights_updator import WeightApdator
from lib.kohonen.normalization.weights_normalization import min_max_normalize

class KohonenNetworkService:
    """Сеть Кохонена с 2D-топологией (карта размера ``rows × cols``).

    Веса хранятся как матрица ``(rows * cols, input_size)``; форма сетки
    необходима для топологического расстояния (см.
    :class:`ITopologicCalculator`) и для визуализации.
    """

    def init_network(
        self,
        rows: int,
        cols: int,
        input_size: int,
    ) -> npt.NDArray[np.float64]:
        if rows <= 0 or cols <= 0 or input_size <= 0:
            raise ValueError(
                f"rows/cols/input_size must be > 0: rows={rows}, cols={cols}, input_size={input_size}"
            )
        return np.random.rand(rows * cols, input_size).astype(np.float64)

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
        normalized = min_max_normalize(input_vector, mins, maxs)
        distances = vector_distance_calc.perform(weights, normalized)
        winner_idx = np.argmin(distances)
        return np.array([float(winner_idx)], dtype=np.float64)

    def get_component_matrix_visualisation(
        self,
        weights: npt.NDArray[np.float64],
        input_id: int,
        samples: npt.NDArray[np.float64],
        rows: int,
        cols: int,
    ) -> npt.NDArray[np.uint8]:
        return get_component_planes_visualisation(
            weights, [], input_id, samples, rows, cols
        )

    def get_u_matrix_visualisation(
        self,
        weights: npt.NDArray[np.float64],
        rows: int,
        cols: int,
    ) -> npt.NDArray[np.uint8]:
        return get_u_matrix_visualisation(weights, rows, cols)
