from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List



class IPerceptron(ABC):
    @abstractmethod
    def save_weights(self): pass

    @abstractmethod
    def load_weights(self, paht: Path): pass

    @abstractmethod
    def get_layers_size(self) -> List[int]: pass

    @abstractmethod
    def set_layer_size(self, layer: int, new_wize:int): pass

    @abstractmethod
    def get_activation_function(self, ): pass

    @abstractmethod
    def set_activation_function(self, new_function: Callable): pass