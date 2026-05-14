import math
import traceback
from typing import Annotated as Annot, Any, Dict, List

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Body, Depends, HTTPException

from config import config, num_constraint_validator as ncv
from lib.kohonen.models.enums import NeighbourhoodFunctionType, TopologyDistanceType
from lib.kohonen.neighbour_function import (
    GaussianNEighborhood,
    INeighbourFunction,
    MexicanHatNeighborhood,
)
from lib.kohonen.normalization import min_max_bounds_from_samples, min_max_normalize
from lib.kohonen.topologic_distance.euclidean import EuclideanTopologicDistance
from lib.kohonen.topologic_distance.manhattan import ManhattanTopologicDistance
from lib.kohonen.vector_distance_calculation.euclidean import EuclideanVectorDistanceCalculator
from models.csv_file import CsvFileData
from models.progect_nn import (
    NNData,
    NNDataWithoutWeights,
    ProjectType,
    ProjectWithData,
    ProjectWithDataWithoutWeights,
)

from container import auth_service, csv_service, kohonen_network_service, project_service
from exceptions.auth_exception import AuthException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from exceptions.not_found import NotFoundException
from log import logger

from ports.api.handlers.tools import oauth2_scheme

router = APIRouter()


def _csv_rows_to_matrix(data: CsvFileData) -> npt.NDArray[np.float64]:
    return np.asarray([row.signs_vector for row in data.rows], dtype=np.float64)


def _map_rows_cols(neurons: int) -> tuple[int, int]:
    r = int(math.isqrt(neurons))
    if r * r != neurons:
        raise ValueError(f"map size must be a perfect square, got {neurons}")
    return r, r


def _neighbourhood(fn: NeighbourhoodFunctionType) -> INeighbourFunction:
    if fn == NeighbourhoodFunctionType.GAUSSIAN:
        return GaussianNEighborhood()
    if fn == NeighbourhoodFunctionType.MEXICAN_HAT:
        return MexicanHatNeighborhood()
    raise ValueError(f"unsupported neighbourhood_function: {fn}")


def _topology_distance(kind: TopologyDistanceType, cols: int):
    if kind == TopologyDistanceType.EUCLIDEAN:
        return EuclideanTopologicDistance(cols=cols)
    if kind == TopologyDistanceType.MANHATTAN:
        return ManhattanTopologicDistance(cols=cols)
    raise ValueError(f"unsupported topology_distance: {kind}")


def _learning_rate_end(learning_rate: float) -> float:
    end = learning_rate * 0.1
    end = max(1e-6, min(end, learning_rate * 0.99))
    if end >= learning_rate:
        end = max(1e-6, learning_rate * 0.5)
    return float(end)


def _sigma_end(sigma_start: float) -> float:
    end = sigma_start * 0.25
    end = max(1e-6, min(end, sigma_start * 0.99))
    if end >= sigma_start:
        end = max(1e-6, sigma_start * 0.5)
    return float(end)


def _weights_matrix(nn_data: NNData) -> npt.NDArray[np.float64]:
    return np.asarray(nn_data.weights[0], dtype=np.float64)


@router.post("/init")
def init_new_kohonen_network(
    token: str = Depends(oauth2_scheme),
    file_id: str = Body(...),
    input_layer_size: Annot[int,ncv(config.PublicConstraints.KOHONEN_INPUT_LAYER_SIZE_RANGE),] = Body(...),
    output_layer_size: Annot[
        int,
        ncv(config.PublicConstraints.KOHONEN_OUTPUT_LAYER_SIZE_RANGE),
    ] = Body(...),
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
        feature_dim = len(data.rows[0].signs_vector)
        if feature_dim != input_layer_size:
            raise HTTPException(
                status_code=400,
                detail=f"CSV feature dim {feature_dim} != input_layer_size {input_layer_size}",
            )

        rows, cols = _map_rows_cols(output_layer_size)
        samples = _csv_rows_to_matrix(data)
        mins, maxs = min_max_bounds_from_samples(samples)

        weights = kohonen_network_service.init_network(
            rows,
            cols,
            mins,
            maxs,
            feature_dim,
        )
        nn_data = NNData(
            weights=[weights.tolist()],
            input_size=feature_dim,
            mins=mins.tolist(),
            maxs=maxs.tolist(),
            classes=data.classes,
        )

        project = project_service.create(
            payload.user_id,
            nn_data=nn_data,
            csv_file_id=file_id,
            project_type=ProjectType.KOHONEN,
        )

        u_img = kohonen_network_service.get_u_matrix_visualisation(
            weights, rows, cols
        )
        primary_image_id = project_service.save_image(
            payload.user_id, project.id, u_img
        )

        component_ids: List[str] = []
        for j in range(feature_dim):
            comp = kohonen_network_service.get_component_matrix_visualisation(
                weights, j, samples, rows, cols
            )
            cid = project_service.save_image(
                payload.user_id, project.id, comp, image_suffix=f"cp_{j}"
            )
            component_ids.append(cid)

    except HTTPException:
        raise
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while initializing kohonen: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "project": ProjectWithDataWithoutWeights(
            id=project.id,
            project_type=ProjectType.KOHONEN,
            user_id=payload.user_id,
            created_at=project.created_at,
            csv_file_id=file_id,
            nn_data=NNDataWithoutWeights(
                input_size=nn_data.input_size,
                mins=nn_data.mins,
                maxs=nn_data.maxs,
                classes=nn_data.classes,
            ),
        ),
        "image_id": primary_image_id,
        "image_ids": {
            "u_matrix": primary_image_id,
            "component_planes": component_ids,
        },
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
) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        p: ProjectWithData = project_service.get_project(payload.user_id, project_id)
        if p.project_type != ProjectType.KOHONEN:
            raise HTTPException(status_code=400, detail="project is not a Kohonen network")
        samples_data: CsvFileData = csv_service.get_data(p.csv_file_id, payload.user_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        if epochs < 1:
            raise HTTPException(status_code=400, detail="epochs must be >= 1")

        weights = _weights_matrix(p.nn_data)
        n_neurons = weights.shape[0]
        rows, cols = _map_rows_cols(n_neurons)

        samples = _csv_rows_to_matrix(samples_data)
        mins = np.asarray(p.nn_data.mins, dtype=np.float64)
        maxs = np.asarray(p.nn_data.maxs, dtype=np.float64)

        nf = _neighbourhood(neighbourhood_function)
        topo = _topology_distance(topology_distance, cols=cols)
        vec = EuclideanVectorDistanceCalculator()

        lr_end = _learning_rate_end(learning_rate)
        sig_end = _sigma_end(initial_neighborhood_radius)

        trained = kohonen_network_service.train(
            weights,
            samples,
            epochs=epochs,
            mins=mins,
            maxs=maxs,
            learning_rate_start=learning_rate,
            learning_rate_end=lr_end,
            sigma_start=initial_neighborhood_radius,
            sigma_end=sig_end,
            vector_distance_calc=vec,
            top_dist_calc=topo,
            neighbour_func=nf,
        )

        updated = NNData(
            weights=[trained.tolist()],
            input_size=p.nn_data.input_size,
            mins=p.nn_data.mins,
            maxs=p.nn_data.maxs,
            classes=p.nn_data.classes,
        )

        u_img = kohonen_network_service.get_u_matrix_visualisation(trained, rows, cols)
        primary_image_id = project_service.save_image(
            payload.user_id, p.id, u_img
        )
        component_ids: List[str] = []
        for j in range(p.nn_data.input_size):
            comp = kohonen_network_service.get_component_matrix_visualisation(
                trained, j, samples, rows, cols
            )
            cid = project_service.save_image(
                payload.user_id, p.id, comp, image_suffix=f"cp_{j}"
            )
            component_ids.append(cid)

        project_service.update_weights(payload.user_id, p.id, updated.weights)
        p.nn_data = updated
    except HTTPException:
        raise
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while learning kohonen: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "project": p.model_dump(),
        "image_id": primary_image_id,
        "image_ids": {
            "u_matrix": primary_image_id,
            "component_planes": component_ids,
        },
    }


@router.post("/get_answer")
def get_answer_kohonen(
    token: str = Depends(oauth2_scheme),
    project_id: str = Body(...),
    input_vector: List[float] = Body(...),
) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        p: ProjectWithData = project_service.get_project(payload.user_id, project_id)
        if p.project_type != ProjectType.KOHONEN:
            raise HTTPException(status_code=400, detail="project is not a Kohonen network")
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        weights = _weights_matrix(p.nn_data)
        n_neurons = weights.shape[0]
        rows, cols = _map_rows_cols(n_neurons)

        mins = np.asarray(p.nn_data.mins, dtype=np.float64)
        maxs = np.asarray(p.nn_data.maxs, dtype=np.float64)
        x = np.asarray(input_vector, dtype=np.float64).ravel()
        if x.size != p.nn_data.input_size:
            raise HTTPException(
                status_code=400,
                detail=f"input_vector length {x.size} != input_size {p.nn_data.input_size}",
            )

        normalized = min_max_normalize(x, mins, maxs)
        vec = EuclideanVectorDistanceCalculator()
        distances = vec.perform(weights, normalized)
        winner = int(np.argmin(distances))
    except HTTPException:
        raise
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while kohonen get_answer: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "winner_neuron_index": winner,
        "map_rows": rows,
        "map_cols": cols,
        "normalized_input": normalized.tolist(),
        "squared_distances": distances.tolist(),
    }
