from typing import List

from log import logger
from activation import IActivation, Rellu
from config import config


def check_forward_propagation_parametrs(inputs: List[float], perceptron: List[List[List[float]]]):
    if(len(inputs) != config.INPUT_LAYER_SIZE):
        e_str = "incorrect size of the array of input values"
        logger.error(e_str)
        raise RuntimeError(e_str)
    

def forward_propogation(inputs: List[float], perceptron: List[List[List[float]]], activation:IActivation):
    check_forward_propagation_parametrs(inputs, perceptron, activation)

    current_activations: List[float] = inputs
    new_activations: List[float] = list()
    sourse_layer_size = config.INPUT_LAYER_SIZE
    target_layer_size = config.HIDDEN_LAYER_SIZE
    for i in range(config.LAYERS_COUNT):

        
        if(i == config.LAYERS_COUNT -1):
            sourse_layer_size = config.HIDDEN_LAYER_SIZE
            target_layer_size = config.OUTPUT_LAYER_SIZE
        elif(i == 1):
            sourse_layer_size = target_layer_size = config.HIDDEN_LAYER_SIZE


        for j in range(target_layer_size):
            current_summ = 0
            for k in range(sourse_layer_size):
                current_summ += current_activations[k]*perceptron[i][j][k] # perceptron [i][j] - не уверен в порядке индексов
            new_activations.append(activation.perform(current_summ))
        current_activations = new_activations

    return current_activations
        