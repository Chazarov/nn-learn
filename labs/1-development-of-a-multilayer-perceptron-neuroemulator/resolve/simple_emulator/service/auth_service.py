import hashlib
import time
import traceback

import jwt

from repository.user_postgres_repository import UserRepository
from models.auth import TokenPayload
from models.user import User
from exceptions.auth_exception import AuthException
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from log import logger


class AuthService:
    def __init__(self, user_repository: UserRepository, jwt_secret: str, jwt_expires_seconds: int):
        self.user_repository = user_repository
        self.jwt_secret = jwt_secret
        self.jwt_expires_seconds = jwt_expires_seconds

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _create_token(self, user_id: str) -> str:
        expired_at = int(time.time()) + self.jwt_expires_seconds
        payload = TokenPayload(user_id = user_id, expired_at=expired_at)
        return jwt.encode(payload.model_dump(), self.jwt_secret, algorithm="HS256")

    def get_token(self, email: str, password: str) -> str:
        try:
            user = self.user_repository.get_user_by_email(email)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting token: {e}")
            traceback.print_exc()
            raise InternalServerException()

        password_hash = self._hash_password(password)
        if user.password_hash != password_hash:
            raise AuthException("Invalid password")

        return self._create_token(user.id)

    def token_validate(self, token: str) -> TokenPayload:
        try:
            decoded = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise AuthException("Token expired")
        except jwt.InvalidTokenError:
            raise AuthException("Invalid token")

        payload = TokenPayload(user_id=decoded["user_id"], expired_at=decoded["expired_at"])

        if payload.expired_at < int(time.time()):
            raise AuthException("Token expired")

        return payload

    def sign_up(self, password: str, email: str, name: str) -> str:
        try:
            hashed_password = self._hash_password(password)
            user: User = self.user_repository.create_user(hashed_password, email, name)
            token = self._create_token(user.id)
            return token
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while signing up: {e}")
            traceback.print_exc()
            raise InternalServerException()
