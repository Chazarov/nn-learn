

import traceback
from typing import List

from pydantic import BaseModel, model_validator

from exceptions.argument_exception import ArgumentException
from nn_logic.models.activation import IActivation
from log import logger


class Perceptron(BaseModel):
    model_config = {'arbitrary_types_allowed': True} 

    weights: List[List[List[float]]]
    activations: List[IActivation]
    layers_count: int

    @model_validator(mode='after')
    def check(self) -> 'Perceptron':
        if(len(self.weights) != (self.layers_count - 1)): 
            logger.error(f"The length of the weights list must be equal to the layers_count-1 value. Weights list length: {len(self.weights)} layers_count value: {self.layers_count}")
            traceback.print_exc()
            raise ArgumentException()
        
        if(len(self.activations) != (self.layers_count - 1)):
            logger.error(f"The length of the activations list must be equal to the layers_count-1 value. Activations list length: {len(self.activations)} layers_count value: {self.layers_count}")
            traceback.print_exc()
            raise ArgumentException()
        
        return self

