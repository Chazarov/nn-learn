import os
import traceback

import numpy as np

from exceptions import DomainException, InternalServerException, NotFoundException
from log import logger
from models.project import NNData


class KohonenDiskRepo:

    def __init__(self, directory: str):
        self.dir = directory

    def create(self, id: str, nn_data: NNData) -> None:
        try:
            os.makedirs(self.dir, exist_ok=True)
            path = os.path.join(self.dir, f"{id}.npz")
            np.savez(
                path,
                weights=nn_data.weights,
                input_size=np.array([nn_data.input_size]),
                mins=nn_data.mins,
                maxs=nn_data.maxs,
                clasters=nn_data.clasters,
            )
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while saving kohonen network: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def delete(self, id: str) -> None:
        try:
            path = os.path.join(self.dir, f"{id}.npz")
            if not os.path.exists(path):
                raise NotFoundException(f"Kohonen network '{id}' not found")
            os.remove(path)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while deleting kohonen network: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_by_id(self, id: str) -> NNData:
        try:
            path = os.path.join(self.dir, f"{id}.npz")
            if not os.path.exists(path):
                raise NotFoundException(f"Kohonen network '{id}' not found")
            data = np.load(path, allow_pickle=False)
            return NNData(
                weights=data["weights"].astype(np.float64),
                input_size=int(data["input_size"].item()),
                mins=data["mins"].astype(np.float64),
                maxs=data["maxs"].astype(np.float64),
                clasters=data["clasters"].astype(np.float64),
            )
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting kohonen network by id: {e}")
            traceback.print_exc()
            raise InternalServerException()