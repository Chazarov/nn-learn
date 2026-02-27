from typing import List

from pydantic import BaseModel


class Project(BaseModel):
    id: str
    user_id: str
    created_at: int
    csv_file_id:str

class NNData(BaseModel):
    weights: List[List[List[float]]]
    input_size: int
    mins: List[float]
    maxs: List[float]
    classes: List[str]

    
class ProjectWithData(Project):
    nn_data:NNData


