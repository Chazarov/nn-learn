from abc import ABC, abstractmethod
from typing import List
from random import randint

from log import logger
from activation import IActivation, Rellu

LAYERS_COUNT = 7

HIDDEN_LAYER_SIZE = 10

INPUT_LAYER_SIZE = 8

OUTPUT_LAYER_SIZE = 4







perceptron = List[List[int]]


def back_propagation():
    pass


def check_forward_propagation_parametrs(inputs: List[float], perceptron: List[List[int[List[int]]]]):
    if(len(inputs) != INPUT_LAYER_SIZE):
        e_str = "incorrect size of the array of input values"
        logger.error(e_str)
        raise RuntimeError(e_str)
    

def forward_propogation(inputs: List[float], perceptron: List[List[int[List[int]]]], activation:IActivation):
    check_forward_propagation_parametrs(inputs, perceptron, activation)

    current_activations: List[float] = inputs
    new_activations: List[float] = list()
    sourse_layer_size = INPUT_LAYER_SIZE
    target_layer_size = HIDDEN_LAYER_SIZE
    for i in range(LAYERS_COUNT):

        
        if(i == LAYERS_COUNT -1):
            sourse_layer_size = HIDDEN_LAYER_SIZE
            target_layer_size = OUTPUT_LAYER_SIZE
        elif(i == 1):
            sourse_layer_size = target_layer_size = HIDDEN_LAYER_SIZE


        for j in range(target_layer_size):
            current_summ = 0
            for k in range(sourse_layer_size):
                current_summ += current_activations[k]*perceptron[i][j][k] # perceptron [i][j] - не уверен в порядке индексов
            new_activations.append(activation.perform(current_summ))
        current_activations = new_activations

    return current_activations
        

def froward_propogation_item(activations):
    pass


def feel_perceptron(perceptron: List[List[int[List[int]]]]):
    for i in range(LAYERS_COUNT - 1):
        if(i == 0):
            perceptron.append(get_randoom_matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))
        elif(i == LAYERS_COUNT - 2):
            perceptron.append(get_randoom_matrix(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE))
        else:
            perceptron.append(get_randoom_matrix(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)) 
    return 

def get_randoom_matrix(n: int, m:int):
    """
    Docstring for get_randoom_matrix
    
    :param n: columns
    :type n: int
    :param m: lines
    :type m: int
    """
    return (randint(100, -100)/100 for i in range(m) for l in range(n))


