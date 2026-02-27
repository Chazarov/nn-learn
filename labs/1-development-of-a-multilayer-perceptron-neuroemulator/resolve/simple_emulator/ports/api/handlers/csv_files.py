from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from exceptions.not_found import NotFoundException
from api.handlers.tools import oauth2_scheme
from container import csv_service, auth_service

router = APIRouter()




@router.post("/upload")
async def upload_csv(file: UploadFile = File(..., description="CSV training sample"),
                    token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")
    payload = auth_service.token_validate(token)
    content: bytes = await file.read()
    result = csv_service.save(payload.user_id, content, file.filename)

    return result.model_dump()


@router.get("/")
async def get_all_samples(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    payload = auth_service.token_validate(token)

    files = csv_service.get_all(user_id=payload.user_id)
    return {
        "files": [file.model_dump() for file in files]
    }


@router.delete("/{file_id}")
async def delete_csv(file_id: str = Query(...),
                    token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    payload = auth_service.token_validate(token)

    try:
        csv_service.delete(payload.user_id, file_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": file_id}
