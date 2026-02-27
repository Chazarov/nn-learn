import uuid
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from exceptions.not_found import NotFoundException
from repository.csv_repository import CsvRepository

router = APIRouter()

_repo: CsvRepository = CsvRepository()


@router.post("/upload")
async def upload_csv(file: UploadFile = File(..., description="CSV training sample")) -> Dict[str, Any]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    file_id: str = str(uuid.uuid4())
    content: bytes = await file.read()
    _repo.save(file_id, content)

    return {"file_id": file_id}


@router.get("/")
async def get_all_samples() -> Dict[str, Any]:
    ids = _repo.get_all()
    return {
        "files": [{"id": file_id, "name": f"{file_id}.csv", "object_type": "file_csv"} for file_id in ids]
    }


@router.delete("/{file_id}")
async def delete_csv(file_id: str) -> Dict[str, Any]:
    try:
        _repo.delete(file_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": file_id}
