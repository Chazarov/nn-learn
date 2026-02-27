from pydantic import BaseModel


class TokenPayload(BaseModel):
    user_id: str
    expired_at: int