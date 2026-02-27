import csv
import json
import os
import random
import uuid
from typing import Any, Dict, List, cast

from fastapi import APIRouter, Body, HTTPException

from activation import ActivationType, ACTIVATIONS
from forwrdpropagation.forward_propagation import forward_propogation
from loss import MSE
from mathh.mv import Sample, apply_adjustments, init_perceptrone as build_perceptrone, normalize
from training.backpropagation import BackPropagation
from repository.image_repository import ImageRepository
from visualisation.visualisation import get_visualisation, ColorTheme

router = APIRouter()

DATA_WEIGHTS = "data/weights"
DATA_LEARN = "data/learn"

_images_repository = ImageRepository()

def _load_csv(path: str) -> tuple[List[Sample], List[str]]:
    rows: List[Sample] = []
    classes: List[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns: List[str] = list(reader.fieldnames or [])
        feature_cols = columns[1:-1]
        label_col = columns[-1]
        for row in reader:
            label = str(row[label_col])
            if label not in classes:
                classes.append(label)
        f.seek(0)
        next(csv.reader(f))
        reader2 = csv.DictReader(open(path, newline=""))
        for row in reader2:
            x: List[float] = [float(row[c]) for c in feature_cols]
            label_idx = classes.index(str(row[label_col]))
            y: List[float] = [1.0 if i == label_idx else 0.0 for i in range(len(classes))]
            rows.append((x, y))
    return rows, classes


@router.post("/init")
def init_new_perceptrone(
    file_id: str = Body(...),
    hidden_layers_architecture: List[int] = Body(...),
) -> Dict[str, Any]:
    path = os.path.join(DATA_LEARN, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, f"File {file_id} not found")

    raw_data, classes = _load_csv(path)

    input_layer_size: int = len(raw_data[0][0])
    output_layer_size: int = len(classes)

    architecture: List[int] = [input_layer_size] + hidden_layers_architecture + [output_layer_size]

    perceptron: List[List[List[float]]] = build_perceptrone(architecture)

    _, mins, maxs = normalize(raw_data)

    perceptron_id: str = str(uuid.uuid4())

    img = get_visualisation(perceptron, ColorTheme.DARK)

    image_id = _images_repository.save_image(perceptron_id, img)

    os.makedirs(DATA_WEIGHTS, exist_ok=True)
    with open(os.path.join(DATA_WEIGHTS, f"{perceptron_id}.json"), "w") as f:
        json.dump({"weights": perceptron, "mins": mins, "maxs": maxs, "classes": classes}, f, ensure_ascii=False, indent=4)

    return {
        "perceptrone_id": perceptron_id,
        "image_id": image_id,
    }


@router.post("/learn/")
def learn_perceptrone(
    file_id: str = Body(...),
    perceptrone_id: str = Body(...),
    activation_type: ActivationType = Body(...),
    epochs: int = Body(...),
    learning_rate: float = Body(...),
) -> Dict[str, Any]:

    path = os.path.join(DATA_LEARN, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, f"File {file_id} not found")

    raw_data, classes = _load_csv(path)


    weights_path: str = os.path.join(DATA_WEIGHTS, f"{perceptrone_id}.json")
    if not os.path.exists(weights_path):
        raise HTTPException(404, f"Perceptrone {perceptrone_id} not found")

    with open(weights_path) as wf:
        saved: Any = json.load(wf)

    perceptron: List[List[List[float]]] = cast(List[List[List[float]]], saved["weights"])
    classes: List[str] = cast(List[str], saved["classes"])

    activation = ACTIVATIONS[activation_type]()
    bp = BackPropagation(MSE(), learning_rate, perceptron, activation)

    data, mins, maxs = normalize(raw_data)
    random.shuffle(data)

    for _ in range(epochs):
        random.shuffle(data)
        for x, y in data:
            outputs, weighted_sums = forward_propogation(x, perceptron, activation)
            adjustments = bp.training_iteration_calculate(x, outputs, y, weighted_sums)
            apply_adjustments(perceptron, adjustments)

    img = get_visualisation(perceptron, ColorTheme.DARK)

    image_id = _images_repository.save_image(perceptrone_id, img)

    os.makedirs(DATA_WEIGHTS, exist_ok=True)
    with open(os.path.join(DATA_WEIGHTS, f"{perceptrone_id}.json"), "w") as f:
        json.dump({"weights": perceptron, "mins": mins, "maxs": maxs, "classes": classes}, f, ensure_ascii=False, indent=4)

    return {
            "perceptrone_id": perceptrone_id,
            "image_id": image_id,
            }


@router.post("/get_answer")
def get_answer(
    perceptrone_id: str = Body(...),
    input_vector: List[float] = Body(...),
    activation_type: ActivationType = Body(...),
) -> Dict[str, Any]:
    path = os.path.join(DATA_WEIGHTS, f"{perceptrone_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Weights {perceptrone_id} not found")

    with open(path) as f:
        saved: Any = json.load(f)

    perceptron = cast(List[List[List[float]]], saved["weights"])
    mins = cast(List[float], saved["mins"])
    maxs = cast(List[float], saved["maxs"])
    classes = cast(List[str], saved["classes"])
    activation = ACTIVATIONS[activation_type]()

    xn: List[float] = [(input_vector[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(input_vector))]
    output_vector, _ = forward_propogation(xn, perceptron, activation)

    predicted: str = classes[output_vector.index(max(output_vector))]
    confidences: Dict[str, float] = {classes[i]: round(output_vector[i], 4) for i in range(len(classes))}

    return {
                "predicted": predicted, 
                "confidences": confidences, 
                "output": output_vector
            }



@router.get("/weights")
async def get_all_weights() -> Dict[str, Any]:
    os.makedirs(DATA_WEIGHTS, exist_ok=True)
    file_names = [n for n in os.listdir(DATA_WEIGHTS) if n.endswith(".json")]
    return {
        "files": [{"id": n.replace(".json", ""), "name": n, "object_type":"file_json"} for n in file_names]
    }


@router.delete("/weights/{perceptrone_id}")
async def delete_weights(perceptrone_id: str) -> Dict[str, Any]:
    path: str = os.path.join(DATA_WEIGHTS, f"{perceptrone_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Weights '{perceptrone_id}' not found")
    os.remove(path)
    return {"deleted": perceptrone_id}
