from pydantic import BaseModel


class User(BaseModel):
    id: str
    password_hash: str
    name: str
    created_at: int
    email: str