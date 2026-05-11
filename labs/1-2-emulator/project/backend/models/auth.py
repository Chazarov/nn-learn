

from pydantic import BaseModel


class SignUpRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str




class TokenPayload(BaseModel):
    user_id: str
    expired_at: int