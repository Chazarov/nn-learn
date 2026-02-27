import csv
import json
import os
import random
import uuid
from typing import Any, Dict, List, cast

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from activation import ActivationType, ACTIVATIONS
from forwrdpropagation.forward_propagation import forward_propogation
from loss import MSE
from mathh.mv import Sample, apply_adjustments, init_perceptrone, normalize
from training.backpropagation import BackPropagation

router = APIRouter(prefix="/api", tags=["API"])

DATA_LEARN = "data/learn"
DATA_WEIGHTS = "data/weights"


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


@router.post("/upload/csv")
async def upload_csv(file: UploadFile = File(..., description="CSV training sample")) -> Dict[str, Any]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    file_id = str(uuid.uuid4())
    dest = os.path.join(DATA_LEARN, f"{file_id}.csv")
    os.makedirs(DATA_LEARN, exist_ok=True)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return {"file_id": file_id}


@router.post("/learn/")
def learn_perceptrone(
    file_id: str = Body(...),
    hidden_layers_architecture: List[int] = Body(...),
    activation_type: ActivationType = Body(...),
    epochs: int = Body(...),
    learning_rate: float = Body(...),
) -> Dict[str, Any]:

    path = os.path.join(DATA_LEARN, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, f"File {file_id} not found")

    raw_data, classes = _load_csv(path)

    input_layer_size: int = len(raw_data[0][0])
    output_layer_size: int = len(classes)

    architecture: List[int] = [input_layer_size] + hidden_layers_architecture + [output_layer_size]

    perceptron = init_perceptrone(architecture)
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

    perceptron_id = str(uuid.uuid4())
    os.makedirs(DATA_WEIGHTS, exist_ok=True)
    with open(os.path.join(DATA_WEIGHTS, f"{perceptron_id}.json"), "w") as f:
        json.dump({"weights": perceptron, "mins": mins, "maxs": maxs, "classes": classes}, f, ensure_ascii=False, indent=4)

    return {"perceptrone_id": perceptron_id}


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

    return {"predicted": predicted, "confidences": confidences, "output": output_vector}


@router.get("/files")
async def get_all_samples() -> Dict[str, Any]:
    os.makedirs(DATA_LEARN, exist_ok=True)
    file_names = [n for n in os.listdir(DATA_LEARN) if n.endswith(".csv")]
    return {
        "files": [{"id": n.replace(".csv", ""), "name": n} for n in file_names]
    }


@router.get("/weights")
async def get_all_weights() -> Dict[str, Any]:
    os.makedirs(DATA_WEIGHTS, exist_ok=True)
    file_names = [n for n in os.listdir(DATA_WEIGHTS) if n.endswith(".json")]
    return {
        "files": [{"id": n.replace(".json", ""), "name": n} for n in file_names]
    }
