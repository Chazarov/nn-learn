import os
import uuid
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile


router = APIRouter()

DATA_LEARN = "data/learn"
DATA_WEIGHTS = "data/weights"





@router.post("/upload")
async def upload_csv(file: UploadFile = File(..., description="CSV training sample")) -> Dict[str, Any]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    file_id = str(uuid.uuid4())
    dest = os.path.join(DATA_LEARN, f"{file_id}.csv")
    os.makedirs(DATA_LEARN, exist_ok=True)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return {"file_id": file_id}


@router.get("/")
async def get_all_samples() -> Dict[str, Any]:
    os.makedirs(DATA_LEARN, exist_ok=True)
    file_names = [n for n in os.listdir(DATA_LEARN) if n.endswith(".csv")]
    return {
        "files": [{"id": n.replace(".csv", ""), "name": n, "object_type":"file_csv"} for n in file_names]
    }