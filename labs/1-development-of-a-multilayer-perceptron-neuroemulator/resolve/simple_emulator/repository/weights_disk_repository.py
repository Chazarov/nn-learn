import json
import os
from typing import Any

from exceptions.not_found import NotFoundException
from models.progect_nn import NNData

class WeightsDiskRepository:

    def __init__(self, directory:str) -> None:
        self.directory = directory


    def create(self, id:str, nn_data: NNData):
        os.makedirs(self.directory, exist_ok=True)
        with open(os.path.join(self.directory, f"{id}.json"), "w") as f:
            json.dump({"weights": nn_data.weights, 
                       "mins": nn_data.mins, 
                       "maxs": nn_data.maxs, 
                       "classes": nn_data.classes}, f, ensure_ascii=False, indent=4)

    def delete(self, id: str) -> None:
        path = os.path.join(self.directory, f"{id}.json")
        if not os.path.exists(path):
            raise NotFoundException(f"Weights file '{id}' not found")
        os.remove(path)

    def get_by_id(self, id:str) -> NNData:
        weights_path: str = os.path.join(self.directory, f"{id}.json")
        if not os.path.exists(weights_path):
            raise NotFoundException(404, f"Perceptrone {id} not found")

        with open(weights_path) as wf:
            saved: Any = json.load(wf)

        return NNData.model_validate(saved)