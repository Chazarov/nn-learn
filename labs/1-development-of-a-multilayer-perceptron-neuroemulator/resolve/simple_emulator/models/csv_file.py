

from typing import List

from pydantic import BaseModel


class CsvFile(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: int


class SampleModel(BaseModel):
    signs_vector: List[float]
    class_mark: List[float]

    
# Where is x - vector of signs, y - class mark
class CsvFileData(BaseModel):
    rows: List[SampleModel]
    classes: List[str]