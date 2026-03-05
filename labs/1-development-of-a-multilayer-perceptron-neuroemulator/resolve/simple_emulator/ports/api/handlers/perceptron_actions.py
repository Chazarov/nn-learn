import traceback
import random
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException, Path

from nn_logic.training.activation import ActivationType, ACTIVATIONS, SoftMax
from nn_logic.forwrdpropagation.forward_propagation import forward_propagation
from nn_logic.loss import LossType, LOSSES, ILoss
from nn_logic.mathh.models import Sample
from nn_logic.mathh.mv import apply_adjustments, init_perceptron as build_perceptron, min_max_samples_normalaize, min_max_signs_normalize
from nn_logic.training.backpropagation import BackPropagation
from nn_logic.visualisation.visualisation import get_visualisation, ColorTheme
from nn_logic.mathh.mv import min_max_samples_normalaize
from models.csv_file import CsvFileData
from models.progect_nn import NNData, ProjectWithData
from nn_logic.mathh.models import Perceptron

from container import csv_service, auth_service, project_service
from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from log import logger

from ports.api.handlers.tools import oauth2_scheme

router = APIRouter()



def _csv_data_to_samples(data: CsvFileData) -> List[Sample]:
    return [Sample(signs=row.signs_vector, class_marks=row.class_mark) for row in data.rows]


@router.post("/init")
def init_new_perceptron(
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
        perceptron: List[List[List[float]]] = build_perceptron(architecture)

        raw_samples: List[Sample] = _csv_data_to_samples(data)
        signs_count = len(raw_samples[0].signs)
        classes_count = len(raw_samples[0].class_marks)
        _, mins, maxs = min_max_samples_normalaize(raw_samples, signs_count=signs_count, classes_count=classes_count)

        nn_data: NNData = NNData(
            weights=perceptron,
            input_size=input_layer_size,
            mins=mins,
            maxs=maxs,
            classes=data.classes,
        )
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
        "project": ProjectWithData(id = project.id, 
                                   user_id = payload.user_id, 
                                   created_at=project.created_at,
                                   csv_file_id=file_id,
                                   nn_data=NNData(weights=perceptron, 
                                                  input_size=nn_data.input_size,
                                                  mins=nn_data.mins,
                                                  maxs=nn_data.maxs,
                                                  classes=nn_data.classes),
                                                  
                                                  ),
        "image_id": image_id,
    }


@router.post("/learn/")
def learn_perceptron(
    token: str = Depends(oauth2_scheme),
    project_id: str = Body(...),
    activation_type: ActivationType = Body(...),
    softmax_use: bool = Body(default=False),
    loss_type: LossType = Body(default=LossType.MSE),
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
        signs_count: int = len(raw_samples[0].signs)
        classes_count: int = len(raw_samples[0].class_marks)


        normalized_samples: List[Sample]
        normalized_samples, _, _ = min_max_samples_normalaize(raw_samples, signs_count=signs_count, classes_count=classes_count)

        layers_count = len(p.nn_data.weights) + 1

        activations = [ACTIVATIONS[activation_type]() for _ in range(layers_count - 1)]
        loss:ILoss = LOSSES[loss_type]
        if(softmax_use):
            activations[-1] = ActivationType.SOFTMAX
        perceptron = Perceptron(weights=p.nn_data.weights, activations=activations, layers_count=layers_count)
        bp = BackPropagation(loss, learning_rate, perceptron)

        for _ in range(epochs):
            random.shuffle(normalized_samples)
            for sample in normalized_samples:
                outputs, weighted_sums = forward_propagation(sample.signs, perceptron)
                adjustments = bp.training_iteration_calculate(sample.signs, outputs, sample.class_marks, weighted_sums)
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
        "project": p.model_dump(),
        "image_id": image_id,
    }


@router.post("/get_answer")
def get_answer(
    token: str = Depends(oauth2_scheme),
    perceptron_id: str = Body(...),
    input_vector: List[float] = Body(...),
    activation_type: ActivationType = Body(...),
    softmax_use: bool = Body(default=False),
) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        p: ProjectWithData = project_service.get_project(payload.user_id, perceptron_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        activations = [ACTIVATIONS[activation_type]()for _ in range(len(p.nn_data.weights))]
        if(softmax_use):
            activations[-1] = SoftMax()
        mins = p.nn_data.mins
        maxs = p.nn_data.maxs
        classes = p.nn_data.classes

        inpupts: List[float] = min_max_signs_normalize(input_vector, maxs=maxs, mins=mins, signs_count=p.nn_data.input_size)

        perceptron: Perceptron = Perceptron(weights=p.nn_data.weights, activations=activations, layers_count=len(p.nn_data.weights))
        output_vector, _ = forward_propagation(inpupts, perceptron)

        
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

@router.get("/project/{project_id}")
def get_project_data(token: str = Depends(oauth2_scheme),
    project_id:str = Path(...)) -> Dict[str, Any]:

    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        project = project_service.get_project(payload.user_id, project_id)
        return {"project": project.model_dump()}
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while getting projects: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/projects/{project_id}")
def delete_project(
    project_id: str = Path(...),
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
