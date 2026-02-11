from pathlib import Path
from typing import Callable, List
import numpy.typing as npt
import numpy as np

from ...exceptions.argument_exception import ArgumentException
from ...logger.log import logger
from .iperceptron import IPerceptron


class Perceptron(IPerceptron):

    __weights: List[npt.NDArray[np.float64]]
    __layers_size: List[int]
    __activation_function: Callable

    def __init__(self, layers_size: List[int], activate: Callable):
        self.__layers_size = layers_size.copy()
        self.__activation_function = activate
        self.__weights = []
        for i in range(len(layers_size) - 1):
            weight_matrix = np.random.randn(layers_size[i] + 1, layers_size[i + 1])
            self.__weights.append(weight_matrix)

    def save_weights(self):
        weights_path = Path("weights.npz")
        np.savez(weights_path, *self.__weights)
        logger.info(f"Weights saved to {weights_path}")

    def load_weights(self, path: Path):
        data = np.load(path)
        self.__weights = [data[key] for key in sorted(data.keys())]
        logger.info(f"Weights loaded from {path}")

    def get_layers_size(self) -> List[int]:
        return self.__layers_size.copy()

    def set_layer_size(self, layer: int, new_size: int):
        self.__set_layer_size(layer, new_size)

    def __set_layer_size(self, layer: int, new_size: int):
        if layer >= len(self.__layers_size) or layer < 0:
            logger.error(f"Incorrect layer index value: {layer}")
            raise ArgumentException("Incorrect layer index value")
        self.__layers_size[layer] = new_size

    def get_activation_function(self):
        return self.__activation_function

    def set_activation_function(self, new_function: Callable):
        self.__set_activation_function(new_function)

    def __set_activation_function(self, new_function: Callable):
        self.__activation_function = new_function