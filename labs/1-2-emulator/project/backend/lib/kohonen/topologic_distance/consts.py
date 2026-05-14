from typing import Dict, Type

from lib.kohonen.models.enums import TopologyDistanceType
from lib.kohonen.topologic_distance.euclidean import EuclideanTopologicDistance
from lib.kohonen.topologic_distance.manhattan import ManhattanTopologicDistance
from lib.kohonen.topologic_distance.base import ITopologicCalculator

TOPOLOGY_CALCULATORS:Dict[str, Type[ITopologicCalculator]] = {
    TopologyDistanceType.EUCLIDEAN.value: EuclideanTopologicDistance,
    TopologyDistanceType.MANHATTAN.value: ManhattanTopologicDistance,
}