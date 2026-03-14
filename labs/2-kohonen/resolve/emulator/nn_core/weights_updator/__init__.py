import numpy.typing as npt
import numpy as np

from exceptions import ArgumentException
from log import logger

from nn_core.topologic_distance import ITopologicCalculator
from nn_core.neighbour_function import INeighbourFunction


class WeightApdator():

    def __init__(self, topologic_distance_calculator:ITopologicCalculator, 
                 neighbour_function:INeighbourFunction, learn_rate: float):
        self.tdc = topologic_distance_calculator
        self.nf = neighbour_function
        self.lr = learn_rate


    def update_weights(self, weights: npt.NDArray[np.float64], learn_rate: float,
                       output_vector: npt.NDArray[np.float64], input_vector: npt.NDArray[np.float64]):
        
        num_clusters, _ = weights.shape 
        sigma:float = 1.0

        if learn_rate > 1 or learn_rate <= 0:
            e_str = f"learn_rate must be in the range 0 < learn_rate <= 1. learn rate: {learn_rate}"
            logger.error(e_str)
            raise ArgumentException(e_str) 

        winner_index = int(np.argmax(output_vector))
        t_dists = self.tdc.perform(winner_index, neurons_count=len(output_vector))
        n_res = self.nf.perform(t_dists, sigma)
        
        for i in range(num_clusters):
            weights[i] += learn_rate * n_res[i] * (input_vector - weights[i])
        
        return weights
        