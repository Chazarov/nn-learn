

from typing import List

from pydantic import BaseModel

from models.base_model import Base


class CsvFile(Base):
    name: str
    is_sample: bool


class SampleModel(BaseModel):
    signs_vector: List[float]
    class_mark: List[float]

    
# Where is x - vector of signs, y - class mark
class CsvFileData(Base):
    rows: List[SampleModel]
    classes: List[str]