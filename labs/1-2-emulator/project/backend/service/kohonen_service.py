import numpy as np
import numpy.typing as npt

from lib.kohonen.decreasing import decreasing_linear_rate, decreasing_linear_sigma
from lib.kohonen.normalization import min_max_normalize, normalize_samples_min_max
from lib.kohonen.neighbour_function import INeighbourFunction
from lib.kohonen.topologic_distance import ITopologicCalculator
from lib.kohonen.vector_distance_calculation import IVectorDistanceCalculator
from lib.kohonen.visualisation import (
    get_component_planes_visualisation,
    get_u_matrix_visualisation,
)
from lib.kohonen.weights_updator import WeightApdator


class KohonenNetworkService:
    """Сеть Кохонена с 2D-топологией (карта размера ``rows × cols``).

    Веса хранятся как матрица ``(rows *cols, input_size)``; форма сетки
    необходима для топологического расстояния (см.
    :class:`ITopologicCalculator`) и для визуализации.
    """

    def init_network(
        self,
        rows: int,
        cols: int,
        mins: npt.NDArray[np.float64],
        maxs: npt.NDArray[np.float64],
        input_size: int,
    ) -> npt.NDArray[np.float64]:
        """Случайные веса ``(rows*cols, input_size)`` в ``[0, 1]`` по каждой компоненте.

        ``mins``/``maxs`` задают ожидаемую размерность ``input_size`` (согласование с
        :meth:`train` / :meth:`predict`); обучение и инференс выполняются в min-max
        нормализованном пространстве с теми же границами.
        """
        if rows <= 0 or cols <= 0 or input_size <= 0:
            raise ValueError(
                f"rows/cols/input_size must be > 0: rows={rows}, cols={cols}, input_size={input_size}"
            )
        lo = np.asarray(mins, dtype=np.float64).ravel()
        hi = np.asarray(maxs, dtype=np.float64).ravel()
        if lo.size != input_size or hi.size != input_size:
            raise ValueError(
                f"mins/maxs length must equal input_size={input_size}, "
                f"got len(mins)={lo.size}, len(maxs)={hi.size}"
            )
        return np.random.rand(rows * cols, input_size).astype(np.float64)

    def train(
        self,
        weights: npt.NDArray[np.float64],
        samples: npt.NDArray[np.float64],
        epochs: int,
        mins: npt.NDArray[np.float64],
        maxs: npt.NDArray[np.float64],
        learning_rate_start: float,
        learning_rate_end: float,
        sigma_start: float,
        sigma_end: float,
        vector_distance_calc: IVectorDistanceCalculator,
        top_dist_calc: ITopologicCalculator,
        neighbour_func: INeighbourFunction,
    ) -> npt.NDArray[np.float64]:
        """
        Обучение SOM: ``epochs`` проходов по выборке; на каждом глобальном шаге скорость
        и ``sigma`` задаются :func:`decreasing_linear_rate` и :func:`decreasing_linear_sigma`
        из ``lib.kohonen.decreasing``. Выборка один раз нормализуется через
        :func:`normalize_samples_min_max` по ``mins``/``maxs`` (построчно то же, что
        :func:`min_max_normalize` в :meth:`predict`).

        **Дополнительно для «полного» цикла (по желанию):** критерий остановки по
        качеству карты, валидация, сохранение чекпоинтов.
        """
        weights = weights.copy()
        if samples.size == 0:
            return weights
        if samples.ndim != 2:
            raise ValueError(
                f"samples must be 2D array (n_samples, n_features), got shape {samples.shape}"
            )
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        dim = samples.shape[1]
        lo = np.asarray(mins, dtype=np.float64).ravel()
        hi = np.asarray(maxs, dtype=np.float64).ravel()
        if lo.size != dim or hi.size != dim:
            raise ValueError(
                f"mins/maxs length must match samples.shape[1]={dim}, "
                f"got len(mins)={lo.size}, len(maxs)={hi.size}"
            )

        n_samples = samples.shape[0]
        total_steps = epochs * n_samples
        normalized_samples = normalize_samples_min_max(samples, lo, hi)
        weight_updator = WeightApdator(
            top_dist_calc, neighbour_func, learning_rate_start
        )
        global_step = 0
        for _ in range(epochs):
            order = np.random.permutation(n_samples)
            for idx in order:
                x = normalized_samples[int(idx)]
                lr = decreasing_linear_rate(
                    global_step, total_steps, learning_rate_start, learning_rate_end
                )
                sigma = decreasing_linear_sigma(
                    global_step, total_steps, sigma_start, sigma_end
                )
                distances = vector_distance_calc.perform(weights, x)
                output_vector = -distances
                weight_updator.update_weights(
                    weights, lr, output_vector, x, sigma
                )
                global_step += 1
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
