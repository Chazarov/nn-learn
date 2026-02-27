

from pydantic import BaseModel


class CsvFile(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: int