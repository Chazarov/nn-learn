import traceback
from typing import List

from sqlalchemy.orm import Session, sessionmaker

from log import logger
from exceptions import InternalServerException, DomainException, NotFoundException
from models.project import KohonenProject
from models.db.kohonen_project import KohonenProjectDB

class KohonenRelativeRepository:
    

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def create(self, user_id: str, csv_file_id: str) -> KohonenProject:
        try:
            with self.session_factory() as session:
                db_project = KohonenProjectDB(user_id=user_id)
                session.add(db_project)
                session.commit()
                session.refresh(db_project)
                return KohonenProject(id=db_project.id, user_id=db_project.user_id,
                               created_at=db_project.created_at, csv_file_id=csv_file_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while creating project: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def delete(self, user_id: str, id: str):
        try:
            with self.session_factory() as session:
                row = session.query(KohonenProjectDB).filter(
                    KohonenProjectDB.id == id, KohonenProjectDB.user_id == user_id
                ).first()
                if row is None:
                    raise NotFoundException(f"Project '{id}' not found")
                session.delete(row)
                session.commit()
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while deleting project: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_by_id(self, user_id: str, id: str) -> KohonenProject:
        try:
            with self.session_factory() as session:
                row = session.query(KohonenProjectDB).filter(
                    KohonenProjectDB.id == id, KohonenProjectDB.user_id == user_id
                ).first()
                if row is None:
                    raise NotFoundException(f"Project '{id}' not found")
                return KohonenProject(id=row.id, user_id=row.user_id,
                               created_at=row.created_at, csv_file_id=row.csv_file_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting project by id: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_all(self, user_id: str) -> List[KohonenProject]:
        try:
            with self.session_factory() as session:
                rows = session.query(KohonenProjectDB).filter(KohonenProjectDB.user_id == user_id).all()
                return [KohonenProject(id=r.id, user_id=r.user_id,
                                created_at=r.created_at, csv_file_id=r.csv_file_id) for r in rows]
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting all projects: {e}")
            traceback.print_exc()
            raise InternalServerException()
