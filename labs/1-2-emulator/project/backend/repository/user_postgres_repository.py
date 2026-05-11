from sqlalchemy.exc import IntegrityError
import traceback

from sqlalchemy.orm import Session, sessionmaker

from models.user import User
from models.db_models import UserDB
from exceptions.not_found import NotFoundException
from exceptions.internal_server_exception import InternalServerException
from exceptions.domain import DomainException
from exceptions.already_exists import AlreadyExists
from log import logger


class UserRepository:

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def get_user(self, id: str) -> User:
        try:
            with self.session_factory() as session:
                row = session.query(UserDB).filter(UserDB.id == id).first()
                if row is None:
                    raise NotFoundException(f"User '{id}' not found")
                return User(id=row.id, password_hash=row.password_hash,
                            name=row.name, created_at=row.created_at, email=row.email)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting user: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def create_user(self, password_hash: str, email: str, name: str) -> User:
        try:
            with self.session_factory() as session:
                db_user = UserDB(password_hash=password_hash, email=email, name=name)
                session.add(db_user)
                session.commit()
                session.refresh(db_user)
                return User(id=db_user.id, password_hash=db_user.password_hash,
                            name=db_user.name, created_at=db_user.created_at, email=db_user.email)
        except DomainException:
            raise
        except IntegrityError as e:
            logger.error(f"user with same data already exists: {e}")
            traceback.print_exc()
            raise AlreadyExists()
        except Exception as e:
            logger.error(f"error while creating user: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_user_by_email(self, email: str) -> User:
        try:
            with self.session_factory() as session:
                row = session.query(UserDB).filter(UserDB.email == email).first()
                if row is None:
                    raise NotFoundException(f"User with email '{email}' not found")
                return User(id=row.id, password_hash=row.password_hash,
                            name=row.name, created_at=row.created_at, email=row.email)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting user by email: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def check_password_by_email(self, email: str, password_hash: str) -> bool:
        try:
            user = self.get_user_by_email(email)
            return user.password_hash == password_hash
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while checking password by email: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def check_password_by_name(self, name: str, password_hash: str) -> bool:
        try:
            with self.session_factory() as session:
                row = session.query(UserDB).filter(UserDB.name == name).first()
                if row is None:
                    raise NotFoundException(f"User '{name}' not found")
                return row.password_hash == password_hash
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while checking password by name: {e}")
            traceback.print_exc()
            raise InternalServerException()
