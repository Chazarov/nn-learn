from models.user import User

class UserRepository:


    def get_user(self, id:str) -> User:
        ...

    def create_user(self, password_hash:str, email:str, name:str) -> User:
        ...

    def check_password_by_email(self, email:str, password_hash:str) -> bool:
        ...

    def check_password_by_name(self, name:str, password_hash:str) -> bool:
        ...

    