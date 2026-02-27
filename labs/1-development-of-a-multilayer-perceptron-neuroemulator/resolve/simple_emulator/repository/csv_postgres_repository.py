from models.csv_file import CsvFile

class CSVPostgresRepository:
    def get_by_user(self, user_id: str):
        ...

    def get_by_id(self, id: str) -> CsvFile:
        ...
