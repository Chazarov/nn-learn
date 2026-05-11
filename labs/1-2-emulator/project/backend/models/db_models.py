import uuid
import time

from sqlalchemy import String, BigInteger, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from base import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


def _now_ts() -> int:
    return int(time.time())


class UserDB(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=_now_ts)


class CsvFileDB(Base):
    __tablename__ = "csv_files"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=_now_ts)
    is_sample: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class ProjectDB(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    csv_file_id: Mapped[str] = mapped_column(String, ForeignKey("csv_files.id"), nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=_now_ts)
