from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from exceptions.argument_exception import ArgumentException





INPUT_LAYER_SIZE = 4
CLUSTERS_COUNT = 5


weights: npt.NDArray[np.float64] = np.random.rand(CLUSTERS_COUNT, INPUT_LAYER_SIZE)



class IDistanceCalculator(ABC):

    @abstractmethod
    def perform(self, weights: npt.NDArray[np.float64], input_vector:npt.NDArray[np.float64]) -> npt.NDArray[np.float64] : pass

class EuclideanDistanceCalculator(IDistanceCalculator):

    def perform(self, weights: npt.NDArray[np.float64], 
                        input_vector:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        

        if(weights.shape[1] != len(input_vector)):
            raise ArgumentException("The number of columns in the weight matrix"
            " is not equal to the size of the input vector. " \
            f"Columns number: {weights.shape[1]}, input vector size: {len(input_vector)}")
        

        ## simple realisation
        # num_clusters, input_size = weights.shape 
        # result: npt.NDArray[np.float64] = np.zeros(weights.shape[0])
        # for i in range(num_clusters):
        #     dist = 0
        #     for j in range(input_size):
        #         dist += (weights[i, j] - input_vector[j])**2
        #     result[i] = dist
        # return result

        return np.sum((weights - input_vector)**2, axis=1)