import traceback
import random
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException

from nn_logic.activation import ActivationType, ACTIVATIONS
from nn_logic.forwrdpropagation.forward_propagation import forward_propogation
from nn_logic.loss import MSE
from nn_logic.mathh.mv import apply_adjustments, init_perceptrone as build_perceptrone, normalize
from nn_logic.training.backpropagation import BackPropagation
from nn_logic.visualisation.visualisation import get_visualisation, ColorTheme
from models.csv_file import CsvFileData
from models.progect_nn import NNData, ProjectWithData

from container import csv_service, auth_service, project_service
from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from log import logger

from api.handlers.tools import oauth2_scheme

router = APIRouter()

Sample = Tuple[List[float], List[float]]


def _csv_data_to_samples(data: CsvFileData) -> List[Sample]:
    return [(row.signs_vector, row.class_mark) for row in data.rows]


@router.post("/init")
def init_new_perceptrone(
    token: str = Depends(oauth2_scheme),
    file_id: str = Body(...),
    hidden_layers_architecture: List[int] = Body(...),
) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        data: CsvFileData = csv_service.get_data(file_id=file_id, user_id=payload.user_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        input_layer_size: int = len(data.rows[0].signs_vector)
        output_layer_size: int = len(data.classes)

        architecture: List[int] = [input_layer_size] + hidden_layers_architecture + [output_layer_size]
        perceptron: List[List[List[float]]] = build_perceptrone(architecture)

        raw_samples: List[Sample] = _csv_data_to_samples(data)
        _, mins, maxs = normalize(raw_samples)

        nn_data: NNData = NNData(weights=perceptron, mins=mins, maxs=maxs, classes=data.classes)
        img = get_visualisation(perceptron, ColorTheme.DARK)

        project = project_service.create(payload.user_id, nn_data=nn_data, csv_file_id=file_id)
        image_id = project_service.save_image(payload.user_id, project.id, img)
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while initializing perceptron: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

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
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        p: ProjectWithData = project_service.get_project(payload.user_id, project_id)
        samples_data: CsvFileData = csv_service.get_data(p.csv_file_id, payload.user_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        raw_samples: List[Sample] = _csv_data_to_samples(samples_data)
        normalized_samples, _, _ = normalize(raw_samples)

        activation = ACTIVATIONS[activation_type]()
        bp = BackPropagation(MSE(), learning_rate, p.nn_data.weights, activation)

        for _ in range(epochs):
            random.shuffle(normalized_samples)
            for x, y in normalized_samples:
                outputs, weighted_sums = forward_propogation(x, p.nn_data.weights, activation)
                adjustments = bp.training_iteration_calculate(x, outputs, y, weighted_sums)
                apply_adjustments(p.nn_data.weights, adjustments)

        img = get_visualisation(p.nn_data.weights, ColorTheme.DARK)
        image_id = project_service.save_image(payload.user_id, p.id, img)
        project_service.update_weights(payload.user_id, p.id, p.nn_data.weights)
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while learning perceptron: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

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
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        p: ProjectWithData = project_service.get_project(payload.user_id, perceptrone_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        activation = ACTIVATIONS[activation_type]()
        mins = p.nn_data.mins
        maxs = p.nn_data.maxs
        classes = p.nn_data.classes

        xn: List[float] = [
            (input_vector[i] - mins[i]) / (maxs[i] - mins[i])
            for i in range(len(input_vector))
        ]
        output_vector, _ = forward_propogation(xn, p.nn_data.weights, activation)

        predicted: str = classes[output_vector.index(max(output_vector))]
        confidences: Dict[str, float] = {
            classes[i]: round(output_vector[i], 4) for i in range(len(classes))
        }
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while getting answer: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "predicted": predicted,
        "confidences": confidences,
        "output": output_vector,
    }


@router.get("/projects")
def get_all_projects(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        projects = project_service.get_projects(payload.user_id)
        return {"projects": [p.model_dump() for p in projects]}
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while getting projects: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/projects/{project_id}")
def delete_project(
    project_id: str,
    token: str = Depends(oauth2_scheme),
) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        project_service.delete_project(payload.user_id, project_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    return {"deleted": project_id}
