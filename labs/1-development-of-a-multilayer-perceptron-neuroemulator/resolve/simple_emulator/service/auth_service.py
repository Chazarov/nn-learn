from repository.user_repository import UserRepository
from models.token_payload import TokenPayload

class AuthService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def get_token(self, user_id:str, password_hash:str)-> str:
        ...



    def token_validate(self, token:str) -> TokenPayload:

        ...


    def sign_up(self, password: str, email:str, name:str) -> User:

        hashed_password = ...
        User:str = self.user_repository.create_user(hashed_password, email, name)
        token = self.get_token(user_id, hashed_password)
        return token
