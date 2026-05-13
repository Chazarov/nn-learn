import traceback
from typing import Annotated, Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException

from config import config, num_constraint_validator as ncv

from lib.kohonen.training_enums import NeighbourhoodFunctionType, TopologyDistanceType
from lib.perceptrone.mathh.models import Sample
from models.csv_file import CsvFileData
from models.progect_nn import NNData, ProjectWithData, ProjectWithDataWithoutWeights

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

# TODO: Cnwert data to kohonen samples


@router.post("/init")
def init_new_kohonen_network(
    token: str = Depends(oauth2_scheme),
    file_id: str = Body(...),
    input_layer_size: Annotated[int, ncv(config.PublicConstraints.KOHONEN_INPUT_LAYER_SIZE_RANGE),] = Body(...),
    output_layer_size: Annotated[int, ncv(config.PublicConstraints.KOHONEN_OUTPUT_LAYER_SIZE_RANGE),] = Body(...),
) -> Dict[str, Any]: 
    return {
        "project": ProjectWithDataWithoutWeights(
            id = project.id, 
            user_id = payload.user_id, 
            created_at=project.created_at, 
            csv_file_id=file_id, 
            nn_data=NNDataWithoutWeights(input_size=nn_data.input_size, mins=nn_data.mins, maxs=nn_data.maxs, classes=nn_data.classes)),
        "image_id": image_id,
    }



@router.post("/learn/")
def learn_kohonen_network(
    token: str = Depends(oauth2_scheme),
    project_id: str = Body(...),
    epochs: int = Body(...),
    learning_rate: float = Body(...),
    initial_neighborhood_radius: float = Body(...),
    neighbourhood_function: NeighbourhoodFunctionType = Body(...),
    topology_distance: TopologyDistanceType = Body(...),
) -> Dict[str, Any]: ...


@router.post("/get_answer")
def get_answer_kohonen(
    token: str = Depends(oauth2_scheme),
    project_id: str = Body(...),
    input_vector: List[float] = Body(...),
) -> Dict[str, Any]: ...
