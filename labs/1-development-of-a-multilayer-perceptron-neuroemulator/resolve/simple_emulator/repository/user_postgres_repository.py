from models.user import User

class UserRepository:


    def get_user(self, id:str) -> User:
        ...

    def create_user(self, password_hash:str, email:str, name:str) -> User:
        ...

    