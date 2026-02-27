import csv
import os
from typing import List

from exceptions.not_found import NotFoundException
from models.csv_file import CsvFileData, SampleModel


class CsvDiskRepository:

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def save(self, file_id: str, content: bytes) -> str:
        os.makedirs(self.directory, exist_ok=True)
        path: str = os.path.join(self.directory, f"{file_id}.csv")
        with open(path, "wb") as f:
            f.write(content)
        return file_id

    def get_all(self) -> List[str]:
        os.makedirs(self.directory, exist_ok=True)
        return [n.replace(".csv", "") for n in os.listdir(self.directory) if n.endswith(".csv")]

    def get_file(self, file_id: str) -> str:
        path: str = os.path.join(self.directory, f"{file_id}.csv")
        if not os.path.exists(path):
            raise NotFoundException(f"CSV file '{file_id}' not found")
        return path

    def delete(self, file_id: str) -> None:
        path: str = os.path.join(self.directory, f"{file_id}.csv")
        if not os.path.exists(path):
            raise NotFoundException(f"CSV file '{file_id}' not found")
        os.remove(path)

    def get_data(self, file_id: str) -> CsvFileData:
        path = self.get_file(file_id)
        rows: List[SampleModel] = []
        classes: List[str] = []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            columns: List[str] = list(reader.fieldnames or [])
            feature_cols = columns[1:-1]
            label_col = columns[-1]

            for row in reader:
                label = str(row[label_col])
                if label not in classes:
                    classes.append(label)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x: List[float] = [float(row[c]) for c in feature_cols]
                label_idx = classes.index(str(row[label_col]))
                y: List[float] = [1.0 if i == label_idx else 0.0 for i in range(len(classes))]
                rows.append(SampleModel(signs_vector=x, class_mark=y))

        return CsvFileData(rows=rows, classes=classes)
