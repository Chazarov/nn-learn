from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from repository.image_repository import ImageRepository
from exceptions.not_found import NotFoundException

router = APIRouter()

_repo: ImageRepository = ImageRepository()


@router.get("/")
async def get_all_images() -> Dict[str, Any]:
    """Получение id и name всех доступных изображений из data/visualisation."""
    ids: List[str] = _repo.get_all_images()
    return {
        "images": [
            {"id": image_id, "name": f"{image_id}.png", "object_type": "image_png"}
            for image_id in ids
        ]
    }


@router.get("/{image_id}")
async def get_image(image_id: str) -> FileResponse:
    """Получение изображения по его id."""
    try:
        path: str = _repo.get_image(image_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(path, media_type="image/png")
