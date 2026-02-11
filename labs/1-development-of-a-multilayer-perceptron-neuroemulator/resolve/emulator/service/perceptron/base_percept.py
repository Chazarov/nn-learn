from pathlib import Path
from typing import Callable, List
import numpy.typing as npt
import numpy as np

from ...exceptions.domain import DomainException 
from .iperceptron import IPerceptron


class Perceptron(IPerceptron):

    __weights:  List[npt.NDArray[np.float64]]
    __layers_size: List[int] = list()

    def __init__(self, layers_size: List[int], activate: Callable): pass

    def save_weights(self): pass

    def load_weights(self, paht: Path): pass

    def get_layers_size(self) -> List[int]: 
        return self.__layers_size.copy()

    def __set_layer_size(self, layer: int, new_size:int):
        if( self.__layers_size <= layer):
            """
            тут должно быть простейшее логгирование этой ошибки а также необходимо
            реализовать класс ArgumentException наследуемый от DomainException
            """
            raise ArgumentException(" Incorrect layer index value ")
        self.__layers_size[layer] = new_size

    def get_activation_function(self): pass

    def __set_activation_function(self, new_function: Callable): pass