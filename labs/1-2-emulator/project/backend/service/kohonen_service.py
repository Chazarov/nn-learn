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
        # Прототипы в [0, 1]^d — та же шкала, что у min-max нормализованных входов.
        return np.random.rand(rows * cols, input_size).astype(np.float64)

    def train(
        self,
        weights: npt.NDArray[np.float64],
        samples: npt.NDArray[np.float64],
        epochs: int,
        mins: npt.NDArray[np.float64],
        maxs: npt.NDArray[np.float64],
        learning_rate: float,
        vector_distance_calc: IVectorDistanceCalculator,
        top_dist_calc: ITopologicCalculator,
        neighbour_func: INeighbourFunction,
    ) -> npt.NDArray[np.float64]:
        """
        Один проход по выборке: для каждого образа вход нормализуется теми же ``mins``/``maxs``,
        что и в :meth:`predict`; обновление весов идёт в этом же пространстве признаков.

        **Что ещё нужно для полноценного цикла обучения SOM (за пределами текущей реализации):**

        1. **Несколько эпох** — многократный проход по обучающей выборке (возможно с
           перемешиванием порядка образов между эпохами).
        2. **Расписание скорости обучения** ``alpha(t)`` — убывание от большего к
           меньшему по мере прогресса (номер эпохи / шаг), а не фиксированная константа.
        3. **Расписание радиуса соседства** ``sigma(t)`` — сужение «колокола»
           Гаусса (или другой функции соседства) по времени; сейчас в ``WeightApdator``
           используется фиксированное значение.
        4. **Согласованная топология** — ``ITopologicCalculator`` с параметром ``cols``;
           число нейронов ``weights.shape[0]`` должно быть кратно ``cols`` (прямоугольная
           решётка ``rows × cols``).
        5. **Согласованная метрика в данных** — тот же ``IVectorDistanceCalculator``, что
           отражает расстояние между входом и весами (здесь — квадрат евклидова расстояния
           в нормализованном пространстве).
        6. **Инициализация весов** — в одной шкале с обучением (например через
           :meth:`init_network` с теми же ``mins``/``maxs``, что и при ``train``/``predict``).
        7. **Границы нормализации** — ``mins``/``maxs`` по обучающей выборке (или по
           train+val); одни и те же при обучении и инференсе.
        8. **Критерий остановки** — максимум эпох, порог изменения весов / ошибки карты,
           ранняя остановка по валидации (по задаче).
        """
        weights = weights.copy()
        if samples.size == 0:
            return weights
        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D array (n_samples, n_features), got shape {samples.shape}")
        dim = samples.shape[1]
        lo = np.asarray(mins, dtype=np.float64).ravel()
        hi = np.asarray(maxs, dtype=np.float64).ravel()
        if lo.size != dim or hi.size != dim:
            raise ValueError(
                f"mins/maxs length must match samples.shape[1]={dim}, "
                f"got len(mins)={lo.size}, len(maxs)={hi.size}"
            )

        weight_updator = WeightApdator(top_dist_calc, neighbour_func, learning_rate)
        for sample in samples:
            x = min_max_normalize(sample, lo, hi)
            distances = vector_distance_calc.perform(weights, x)
            output_vector = -distances
            weight_updator.update_weights(weights, learning_rate, output_vector, x)
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
