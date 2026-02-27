import traceback
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from api.handlers.tools import oauth2_scheme
from container import csv_service, auth_service
from log import logger

router = APIRouter()


@router.post("/upload")
async def upload_csv(file: UploadFile = File(..., description="CSV training sample"),
                     token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        content: bytes = await file.read()
        result = csv_service.save(payload.user_id, content, file.filename)
        return result.model_dump()
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while uploading csv: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def get_all_samples(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        files = csv_service.get_all(user_id=payload.user_id)
        return {"files": [file.model_dump() for file in files]}
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while getting csv files: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{file_id}")
async def delete_csv(file_id: str = Query(...),
                     token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        csv_service.delete(payload.user_id, file_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while deleting csv: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

    return {"deleted": file_id}
