from random import randint
from typing import List
from config import config



def feel_perceptron(perceptron: List[List[List[float]]]):
    for i in range(config.LAYERS_COUNT - 1):
        if(i == 0):
            perceptron.append(get_randoom_matrix(config.INPUT_LAYER_SIZE, config.HIDDEN_LAYER_SIZE))
        elif(i == config.LAYERS_COUNT - 2):
            perceptron.append(get_randoom_matrix(config.HIDDEN_LAYER_SIZE, config.OUTPUT_LAYER_SIZE))
        else:
            perceptron.append(get_randoom_matrix(config.HIDDEN_LAYER_SIZE, config.HIDDEN_LAYER_SIZE)) 
    return perceptron



def get_randoom_matrix(n: int, m:int):
    """
    Docstring for get_randoom_matrix
    
    :param n: columns
    :type n: int
    :param m: lines
    :type m: int
    """
    return (randint(100, -100)/100 for i in range(m) for l in range(n))


