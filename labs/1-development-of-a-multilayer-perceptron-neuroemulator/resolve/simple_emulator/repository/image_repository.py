import os
from typing import List

import cv2
import numpy as np
import numpy.typing as npt

from exceptions.not_found import NotFoundException

class ImageRepository:

    DIRECTORY:str = "data/visualisation"

    def save_image(self, object_id: str, image:npt.NDArray[np.uint8])-> str:

        out_path: str = os.path.join(self.DIRECTORY, f"{object_id}.png")
        os.makedirs(self.DIRECTORY, exist_ok=True)  # создаст папку если не существует
        cv2.imwrite(out_path, image)

        print(f"Image saved: {out_path}  shape={image.shape}")

        return object_id


    def get_all_images(self) -> List[str]:
        os.makedirs(self.DIRECTORY, exist_ok=True)
        file_names: List[str] = [n for n in os.listdir(self.DIRECTORY) if n.endswith(".png")]
        return [n.replace(".png", "") for n in file_names]


    def get_image(self, image_id: str) -> str:
        path: str = os.path.join(self.DIRECTORY, f"{image_id}.png")
        if not os.path.exists(path):
            raise NotFoundException(f"Image '{image_id}' not found")
        return path
