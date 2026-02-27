import json
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import numpy.typing as npt

from nn_logic.visualisation.visualisation import get_visualisation, ColorTheme


def test_visualisation_iris() -> None:
    data_path: str = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "weights", "sample_Iris.json"
    )

    with open(data_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    perceptrone: List[List[List[float]]] = data["weights"]

    img: npt.NDArray[np.uint8] = get_visualisation(perceptrone, ColorTheme.DARK)

    out_path: str = os.path.join(os.path.dirname(__file__), "visualisation_example.png")
    cv2.imwrite(out_path, img)

    print(f"Image saved: {out_path}  shape={img.shape}")
