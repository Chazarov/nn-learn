from typing import List

from models.csv_file import CsvFile

class CSVRelativeRepository:
    def create(self, user_id:str, name:str) -> CsvFile:
        ...

    def get_by_user(self, user_id: str) -> List[CsvFile]:
        ...

    def get_by_id(self, id: str, user_id:str) -> CsvFile:
        ...

    def delete(self, id: str, user_id: str):
        ...
