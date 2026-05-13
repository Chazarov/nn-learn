import traceback

from fastapi import APIRouter, Depends, HTTPException, Path


from container import auth_service, project_service
from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from log import logger

from ports.api.handlers.tools import oauth2_scheme

router = APIRouter()







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
