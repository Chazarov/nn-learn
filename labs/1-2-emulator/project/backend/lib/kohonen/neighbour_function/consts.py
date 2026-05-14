

from ast import Dict
from typing import Type
from lib.kohonen.models.enums import NeighbourhoodFunctionType
from lib.kohonen.neighbour_function.base import INeighbourFunction
from lib.kohonen.neighbour_function import GaussianNEighborhood, MexicanHatNeighborhood


NEIGHBOURHOOD_FUNCTIONS:Dict[str, Type[INeighbourFunction]] = {
    NeighbourhoodFunctionType.GAUSSIAN.value: GaussianNEighborhood,
    NeighbourhoodFunctionType.MEXICAN_HAT.value: MexicanHatNeighborhood,
}