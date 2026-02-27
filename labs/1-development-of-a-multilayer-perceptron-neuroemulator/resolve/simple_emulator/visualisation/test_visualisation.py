import json
import os
import cv2

from .visualisation import get_visualisation


def test_visualisation_iris():
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "weights", "sample_Iris.json"
    )

    with open(data_path, "r") as f:
        data = json.load(f)

    perceptrone = data["weights"]

    img = get_visualisation(perceptrone)

    out_path = os.path.join(os.path.dirname(__file__), "visualisation_example.png")
    cv2.imwrite(out_path, img)

    print(f"Image saved: {out_path}  shape={img.shape}")
