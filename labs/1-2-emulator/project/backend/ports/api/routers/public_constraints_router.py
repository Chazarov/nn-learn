from typing import Any, Dict

from fastapi import APIRouter

from ports.api.handlers.public_constraints_handler import build_public_constraints_json

router = APIRouter()


@router.get("")
def get_public_constraints() -> Dict[str, Any]:
    return build_public_constraints_json()
