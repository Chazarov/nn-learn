from fastapi import APIRouter,  Depends, HTTPException, Query
from fastapi.responses import FileResponse

from exceptions.not_found import NotFoundException, DomainException
from models.token_payload import TokenPayload
from container import project_service
from container import auth_service


from api.handlers.tools import oauth2_scheme

router = APIRouter()





@router.get("/{image_id}")
async def get_image(image_id: str = Query(...),
                    token: str = Depends(oauth2_scheme)) -> FileResponse:
    """Получение изображения по его id."""
    try:
        payload:TokenPayload = auth_service.token_validate(token)
    except DomainException: raise
    except:
        # В модуле exceptions разработать исключения для ошибок связанных с токеном здесь их все обработать
        # а в auth_service сделать логику выбрасывания этих шибок п валидации 
        return

    try:
        path: str = project_service.get_image(project_id=image_id, user_id = payload.user_id)
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(path, media_type="image/png")
