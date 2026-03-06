import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException, Path

from nn_logic.models.activation import ActivationType
from nn_logic.loss import LossType
from nn_logic.mathh.models import Sample
from models.csv_file import CsvFileData
from models.progect_nn import NNData, ProjectWithData

from container import csv_service, auth_service, project_service, nn_service
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
        raw_samples: List[Sample] = _csv_data_to_samples(data)

        weights, mins, maxs = nn_service.init_perceptron(architecture, raw_samples)

        nn_data: NNData = NNData(
            weights=weights,
            input_size=input_layer_size,
            mins=mins,
            maxs=maxs,
            classes=data.classes,
        )
        img = nn_service.get_visualisation(weights)

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
                                   nn_data=NNData(weights=weights, 
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

        p.nn_data.weights = nn_service.train(
            weights=p.nn_data.weights,
            samples=raw_samples,
            activation_type=activation_type,
            loss_type=loss_type,
            softmax_use=softmax_use,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        img = nn_service.get_visualisation(p.nn_data.weights)
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
        classes = p.nn_data.classes

        output_vector = nn_service.predict(
            weights=p.nn_data.weights,
            input_vector=input_vector,
            activation_type=activation_type,
            softmax_use=softmax_use,
            mins=p.nn_data.mins,
            maxs=p.nn_data.maxs,
            input_size=p.nn_data.input_size,
        )

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
