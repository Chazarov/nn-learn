import json
import os
import random
import uuid
from typing import Any, Dict, List, cast


from fastapi import APIRouter, Body, Depends, HTTPException



from nn_logic.activation import ActivationType, ACTIVATIONS
from nn_logic.forwrdpropagation.forward_propagation import forward_propogation
from nn_logic.loss import MSE
from nn_logic.mathh.mv import apply_adjustments, init_perceptrone as build_perceptrone, normalize
from nn_logic.training.backpropagation import BackPropagation
from nn_logic.visualisation.visualisation import get_visualisation, ColorTheme
from models.csv_file import CsvFileData
from models.progect_nn import NNData, Project, ProjectWithData

from container import csv_service, auth_service, project_service

from api.handlers.tools import oauth2_scheme



router = APIRouter()

DATA_WEIGHTS = "data/weights"
DATA_LEARN = "data/learn"



@router.post("/init")
def init_new_perceptrone(
    token: str = Depends(oauth2_scheme),
    file_id: str = Body(...),
    hidden_layers_architecture: List[int] = Body(...),
) -> Dict[str, Any]:
    path = os.path.join(DATA_LEARN, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, f"File {file_id} not found")

    payload = auth_service.token_validate(token)

    data:CsvFileData = csv_service.get_data(file_id=file_id, user_id=payload.user_id)

    input_layer_size: int = len(data.rows[0].signs_vector)
    output_layer_size: int = len(data.classes)

    architecture: List[int] = [input_layer_size] + hidden_layers_architecture + [output_layer_size]

    perceptron: List[List[List[float]]] = build_perceptrone(architecture)

    _, mins, maxs = normalize(raw_data) # Исправить процесс инициализации 

    nn_data :NNData = NNData(weights=perceptron, mins=mins, maxs=maxs, classes = data.classes)


    img = get_visualisation(perceptron, ColorTheme.DARK)

    project = project_service.create(payload.user_id, nn_data=nn_data, csv_file_id=file_id)
    image_id = project_service.save_image(payload.user_id, project.id, img)


    return {
        "perceptrone_id": project.id,
        "image_id": image_id,
    }


@router.post("/learn/")
def learn_perceptrone(
    token: str = Depends(oauth2_scheme),
    project_id: str = Body(...),
    activation_type: ActivationType = Body(...),
    epochs: int = Body(...),
    learning_rate: float = Body(...),
) -> Dict[str, Any]:

    payload = auth_service.token_validate(token)

    p:ProjectWithData = project_service.get_project(payload.user_id, project_id)
    samples_data: CsvFileData = csv_service.get_data(p.csv_file_id, payload.user_id)
    

    activation = ACTIVATIONS[activation_type]()
    bp = BackPropagation(MSE(), learning_rate, p.nn_data.weights, activation)


    # Здесь исправить обучение. Оно должно быть по данным из полученных схем (samples_data)
    random.shuffle(data)

    for _ in range(epochs):
        random.shuffle(data)
        for x, y in data:
            outputs, weighted_sums = forward_propogation(x, p.nn_data.weights, activation)
            adjustments = bp.training_iteration_calculate(x, outputs, y, weighted_sums)
            apply_adjustments(p.nn_data.weights, adjustments)

    img = get_visualisation(p.nn_data.weights, ColorTheme.DARK)



    image_id = project_service.save_image(payload.user_id, p.id, img)
    project_service.update_weights(payload.user_id, p.id, p.nn_data.weights) #Этот метод не законен. Закончи его реализацию


    return {
            "project_id": p.id,
            "image_id": image_id,
            }


@router.post("/get_answer")
def get_answer(
    token: str = Depends(oauth2_scheme),
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
async def get_all_weights(token: str = Depends(oauth2_scheme),) -> Dict[str, Any]:
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
