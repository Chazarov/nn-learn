from typing import Any, Dict

from pydantic import BaseModel

from config import config


def build_public_constraints_json() -> Dict[str, Any]:
    """Serialize ``Config.PublicConstraints`` for the public API (nested models as dicts)."""
    out: Dict[str, Any] = {}
    for name, value in vars(config.PublicConstraints).items():
        if name.startswith("_"):
            continue
        if isinstance(value, BaseModel):
            out[name] = value.model_dump()
        else:
            out[name] = value
    return out
