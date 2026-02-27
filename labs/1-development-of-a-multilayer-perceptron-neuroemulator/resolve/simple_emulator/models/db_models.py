import uuid
import time

from sqlalchemy import Column, String, BigInteger, ForeignKey
from database import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())

def _now_ts() -> int:
    return int(time.time())


class UserDB(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_generate_uuid)
    password_hash = Column(String, nullable=False)
    name = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    created_at = Column(BigInteger, nullable=False, default=_now_ts)


class CsvFileDB(Base):
    __tablename__ = "csv_files"

    id = Column(String, primary_key=True, default=_generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(BigInteger, nullable=False, default=_now_ts)


class ProjectDB(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    csv_file_id = Column(String, ForeignKey("csv_files.id"), nullable=False)
    created_at = Column(BigInteger, nullable=False, default=_now_ts)
