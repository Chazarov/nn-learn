
from typing import List


from training.itraining_algorithm import ITrainingAlgorithm
from .update_weights_algorithm import IUpdatingWeightsAlgorithm
from ..loss import ILoss



class BackPropagation(ITrainingAlgorithm):

    updw_alg : IUpdatingWeightsAlgorithm
    perceptrone: List[List[List[float]]]
    loss: ILoss

    def __init__(self, loss: ILoss, update_weights_algorithm: IUpdatingWeightsAlgorithm, perceptrone: List[List[List[float]]]):
        self.updw_alg = update_weights_algorithm
        self.perceptrone = perceptrone
        self.loss = loss

    def train(self, inputs: List[float], loss: float):


        pass

    def get_perceptrone(self): return self.perceptrone

    def get_loos_function(self): return self.loss


    