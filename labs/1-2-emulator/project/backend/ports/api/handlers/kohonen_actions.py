import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException, Path

from lib.perceptrone.models.activation import ActivationType
from lib.perceptrone.loss import LossType
from lib.perceptrone.mathh.models import Sample
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

# TODO: Cnwert data to kohonen samples


@router.post("/init")
def init_new_kohonen_network(
    token: str = Depends(oauth2_scheme),
    file_id: str = Body(...),
    input_layer_size: int = Body(...),
    output_layer_size: int = Body(...)
) -> Dict[str, Any]: ... 


@router.post("/learn/")
def learn_kohonen_network(
    token: str = Depends(oauth2_scheme),
    project_id: str = Body(...),
    activation_type: ActivationType = Body(...),
    softmax_use: bool = Body(default=False),
    loss_type: LossType = Body(default=LossType.MSE),
    epochs: int = Body(...),
    learning_rate: float = Body(...),
) -> Dict[str, Any]: ...


@router.post("/get_answer")
def get_answer_kohonen(
    token: str = Depends(oauth2_scheme),
    perceptron_id: str = Body(...),
    input_vector: List[float] = Body(...),
    activation_type: ActivationType = Body(...),
    softmax_use: bool = Body(default=False),
) -> Dict[str, Any]: ...
