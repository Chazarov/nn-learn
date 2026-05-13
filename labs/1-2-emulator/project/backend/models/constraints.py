from pydantic import BaseModel
from typing import Optional, List

class NumConstraint(BaseModel):
    """If allowed_values is not None, only these values are allowed; otherwise, check by min_value..max_value."""

    name: str
    min_value: int = 0
    max_value: int = 100_000
    allowed_values: Optional[List[int]] = None