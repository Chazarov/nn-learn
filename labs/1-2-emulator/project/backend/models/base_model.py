


from pydantic import BaseModel


class Base(BaseModel):
    id: str
    user_id: str
    created_at: int