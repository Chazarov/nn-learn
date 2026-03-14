import numpy.typing as npt
import numpy as np

from nn_core.vector_distance_calculation import IVectorDistanceCalculator
from exceptions import ArgumentException
from log import logger


class EuclideanVectorDistanceCalculator(IVectorDistanceCalculator):

    def perform(self, weights: npt.NDArray[np.float64], 
                        input_vector:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        

        if(weights.shape[1] != len(input_vector)):
            e_str = "The number of columns in the weight matrix"
            " is not equal to the size of the input vector. " \
            f"Columns number: {weights.shape[1]}, input vector size: {len(input_vector)}"
            logger.error(e_str)
            raise ArgumentException()
        

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