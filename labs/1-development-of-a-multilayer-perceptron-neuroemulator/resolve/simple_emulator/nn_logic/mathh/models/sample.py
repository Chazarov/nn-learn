

from typing import List

from pydantic import BaseModel


class Sample(BaseModel):
    signs: List[float]
    class_marks: List[float]