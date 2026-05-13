from enum import Enum


class NeighbourhoodFunctionType(str, Enum):
    """Вид функции соседства h_cj(t) (см. шаг настройки в методичке по SOM)."""

    GAUSSIAN = "GAUSSIAN"
    MEXICAN_HAT = "MEXICAN_HAT"


class TopologyDistanceType(str, Enum):
    """Метрика топологического расстояния между узлами на карте."""

    EUCLIDEAN = "EUCLIDEAN"
    MANHATTAN = "MANHATTAN"
