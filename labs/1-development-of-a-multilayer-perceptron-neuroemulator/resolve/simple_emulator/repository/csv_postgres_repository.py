import traceback
from typing import List

from sqlalchemy.orm import Session, sessionmaker

from models.csv_file import CsvFile
from models.db_models import CsvFileDB
from exceptions.not_found import NotFoundException
from exceptions.internal_server_exception import InternalServerException
from exceptions.domain import DomainException
from log import logger


class CSVRelativeRepository:

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def create(self, user_id: str, name: str) -> CsvFile:
        try:
            with self.session_factory() as session:
                db_file = CsvFileDB(user_id=user_id, name=name)
                session.add(db_file)
                session.commit()
                session.refresh(db_file)
                return CsvFile(id=db_file.id, user_id=db_file.user_id,
                               name=db_file.name, created_at=db_file.created_at)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while creating csv file record: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_by_user(self, user_id: str) -> List[CsvFile]:
        try:
            with self.session_factory() as session:
                rows = session.query(CsvFileDB).filter(CsvFileDB.user_id == user_id).all()
                return [CsvFile(id=r.id, user_id=r.user_id,
                                name=r.name, created_at=r.created_at) for r in rows]
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting csv files by user: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_by_id(self, id: str, user_id: str) -> CsvFile:
        try:
            with self.session_factory() as session:
                row = session.query(CsvFileDB).filter(
                    CsvFileDB.id == id, CsvFileDB.user_id == user_id
                ).first()
                if row is None:
                    raise NotFoundException(f"CSV file '{id}' not found")
                return CsvFile(id=row.id, user_id=row.user_id,
                               name=row.name, created_at=row.created_at)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting csv file by id: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def delete(self, id: str, user_id: str):
        try:
            with self.session_factory() as session:
                row = session.query(CsvFileDB).filter(
                    CsvFileDB.id == id, CsvFileDB.user_id == user_id
                ).first()
                if row is None:
                    raise NotFoundException(f"CSV file '{id}' not found for this user")
                session.delete(row)
                session.commit()
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while deleting csv file record: {e}")
            traceback.print_exc()
            raise InternalServerException()
