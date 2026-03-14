from pydantic import BaseModel, Field
import numpy as np
import numpy.typing as npt
from models.base_model import Base


class KohonenProject(Base):
    csv_file_id:str

class NNData(BaseModel):
    weights: npt.NDArray[np.float64] = Field(exclude=True)
    input_size: int
    mins: npt.NDArray[np.float64]
    maxs: npt.NDArray[np.float64]
    clasters: npt.NDArray[np.float64]

    
class ProjectWithData(KohonenProject):
    nn_data:NNData
