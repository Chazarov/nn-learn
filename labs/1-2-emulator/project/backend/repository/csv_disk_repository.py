import csv
import os
from typing import List, Optional, Sequence, Tuple

from exceptions.domain import DomainException
from exceptions.not_found import NotFoundException
from models.csv_file import CsvFileData, SampleModel


def _non_empty_fieldnames(fieldnames: Optional[Sequence[str]]) -> List[str]:
    """Порядок ключей DictReader без пустых имён (часто из‑за лишней запятой в заголовке Excel)."""
    out: List[str] = []
    for raw in fieldnames or []:
        if raw is None:
            continue
        if not str(raw).strip():
            continue
        out.append(str(raw))
    return out


def _feature_and_label_keys(fieldnames: Optional[Sequence[str]]) -> Tuple[List[str], str]:
    """Формат: ``номер_примера, признак1, …, признакK, метка`` — первая колонка не признак, метка всегда последняя."""
    keys = _non_empty_fieldnames(fieldnames)
    if len(keys) < 3:
        raise DomainException(
            "В CSV нужно минимум три колонки: номер примера, хотя бы один признак и метка в последней колонке",
        )
    label_key = keys[-1]
    feature_keys = keys[1:-1]
    if not feature_keys:
        raise DomainException(
            "Нет колонок признаков между номером примера и меткой класса",
        )
    return feature_keys, label_key


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

        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            raw_dict_rows: List[dict] = list(reader)

        feature_cols, label_col = _feature_and_label_keys(fieldnames)

        for row in raw_dict_rows:
            label = str(row[label_col])
            if label not in classes:
                classes.append(label)

        for row in raw_dict_rows:
            try:
                x = [float(row[c]) for c in feature_cols]
            except (TypeError, ValueError) as e:
                raise DomainException(
                    "Ожидаются числовые значения во всех колонках признаков; "
                    f"проверьте строку и заголовки ({e})",
                ) from e
            label_idx = classes.index(str(row[label_col]))
            y: List[float] = [1.0 if i == label_idx else 0.0 for i in range(len(classes))]
            rows.append(SampleModel(signs_vector=x, class_mark=y))

        return CsvFileData(rows=rows, classes=classes)
    