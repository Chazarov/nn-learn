import traceback
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from container import auth_service, csv_service
from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from exceptions.already_exists import AlreadyExists
from models.auth import SignUpRequest, LoginRequest
from log import logger
from ports.api.handlers.tools import oauth2_scheme

router = APIRouter()




@router.post("/sign-up")
def sign_up(body: SignUpRequest) -> Dict[str, Any]:
    try:
        token = auth_service.sign_up(
            password=body.password, email=body.email, name=body.name
        )
        pl = auth_service.token_validate(token)
        csv_service.init_sample(pl.user_id)
        return {"token": token}
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")
    except AlreadyExists as e:
        raise HTTPException(status_code=403, detail=str("user with same data already exists"))
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error in sign-up handler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/login")
def login(body: LoginRequest) -> Dict[str, Any]:
    try:
        token = auth_service.get_token(email=body.email, password=body.password)
        return {"token": token}
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error in login handler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/getme")
def getme(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        pl = auth_service.token_validate(token)
        user = auth_service.get_me(pl.user_id)
        return {"me": user}
    except AuthException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InternalServerException:
        raise HTTPException(status_code=500, detail="Internal server error")
    except DomainException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"error in login handler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
