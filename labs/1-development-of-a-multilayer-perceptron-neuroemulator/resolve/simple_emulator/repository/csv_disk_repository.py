import os
from typing import List

from exceptions.not_found import NotFoundException


class CsvRepository:

    DIRECTORY: str = "data/learn"

    def save(self, file_id: str, content: bytes) -> str:
        os.makedirs(self.DIRECTORY, exist_ok=True)
        path: str = os.path.join(self.DIRECTORY, f"{file_id}.csv")
        with open(path, "wb") as f:
            f.write(content)
        return file_id

    def get_all(self) -> List[str]:
        os.makedirs(self.DIRECTORY, exist_ok=True)
        return [n.replace(".csv", "") for n in os.listdir(self.DIRECTORY) if n.endswith(".csv")]

    def get_path(self, file_id: str) -> str:
        path: str = os.path.join(self.DIRECTORY, f"{file_id}.csv")
        if not os.path.exists(path):
            raise NotFoundException(f"CSV file '{file_id}' not found")
        return path

    def delete(self, file_id: str) -> None:
        path: str = os.path.join(self.DIRECTORY, f"{file_id}.csv")
        if not os.path.exists(path):
            raise NotFoundException(f"CSV file '{file_id}' not found")
        os.remove(path)
