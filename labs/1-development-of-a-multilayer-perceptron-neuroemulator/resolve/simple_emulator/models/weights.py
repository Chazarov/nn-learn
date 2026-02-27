from typing import List

from pydantic import BaseModel


class WeightsMeta(BaseModel):
    id: str
    user_id: str
    created_at: int

class WeightsData(BaseModel):
    weights: List[List[List[float]]]
    mins: List[float]
    maxs: List[float]
    classes: List[str]