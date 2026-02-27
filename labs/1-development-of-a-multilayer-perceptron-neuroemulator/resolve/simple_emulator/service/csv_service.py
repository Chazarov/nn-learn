import traceback
from typing import List

from models.csv_file import CsvFile, CsvFileData
from repository.csv_disk_repository import CsvDiskRepository
from repository.csv_postgres_repository import CSVRelativeRepository
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from log import logger


class CsvService:
    def __init__(self, disk_repo: CsvDiskRepository, relative_repo: CSVRelativeRepository):
        self.disk_repo = disk_repo
        self.relative_repo = relative_repo

    def save(self, user_id: str, content: bytes, name: str) -> CsvFile:
        try:
            file: CsvFile = self.relative_repo.create(user_id, name)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while saving csv file metadata: {e}")
            traceback.print_exc()
            raise InternalServerException()

        try:
            self.disk_repo.save(file.id, content)
        except DomainException:
            self.relative_repo.delete(file.id, user_id)
            raise
        except Exception as e:
            self.relative_repo.delete(file.id, user_id)
            logger.error(f"error while saving csv file to disk: {e}")
            traceback.print_exc()
            raise InternalServerException()

        return file

    def get_all(self, user_id: str) -> List[CsvFile]:
        return self.relative_repo.get_by_user(user_id)

    def delete(self, user_id: str, file_id: str):
        self.relative_repo.delete(file_id, user_id)
        self.disk_repo.delete(file_id)

    def get_data(self, file_id: str, user_id: str) -> CsvFileData:
        self.relative_repo.get_by_id(file_id, user_id)
        return self.disk_repo.get_data(file_id)
