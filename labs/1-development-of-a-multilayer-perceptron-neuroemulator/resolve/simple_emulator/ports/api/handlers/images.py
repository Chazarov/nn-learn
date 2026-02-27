import traceback

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import FileResponse

from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from models.auth import TokenPayload
from container import project_service, auth_service
from log import logger

from ports.api.handlers.tools import oauth2_scheme

router = APIRouter()


@router.get("/{image_id}")
async def get_image(image_id: str = Path(),
                    token: str = Depends(oauth2_scheme)) -> FileResponse:
    try:
        payload: TokenPayload = auth_service.token_validate(token)
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        path: str = project_service.get_image(project_id=image_id, user_id=payload.user_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error while getting image: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

    return FileResponse(path, media_type="image/png")
