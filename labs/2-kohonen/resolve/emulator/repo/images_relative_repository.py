import traceback

from sqlalchemy.orm import Session, sessionmaker

from models.image import Image
from models.db.image import ImageDB
from exceptions.not_found import NotFoundException
from exceptions import InternalServerException, DomainException
from log import logger


class ImageRelativeRepository:

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def create(self, user_id: str, name: str) -> Image:
        try:
            with self.session_factory() as session:
                db_image = ImageDB(user_id=user_id, name=name)
                session.add(db_image)
                session.commit()
                session.refresh(db_image)
                return Image(id=db_image.id, user_id=db_image.user_id,
                               name=db_image.name, created_at=db_image.created_at)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while creating image file record: {e}")
            traceback.print_exc()
            raise InternalServerException()


    def get_by_id(self, id: str, user_id: str) -> Image:
        try:
            with self.session_factory() as session:
                row = session.query(ImageDB).filter(
                    ImageDB.id == id, ImageDB.user_id == user_id
                ).first()
                if row is None:
                    raise NotFoundException(f"image with id: '{id}' not found")
                return Image(id=row.id, user_id=row.user_id,
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
                row = session.query(ImageDB).filter(
                    ImageDB.id == id, ImageDB.user_id == user_id
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
