from sqlalchemy import String, BigInteger, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from base import Base, now_ts, generate_uuid

class CsvFileDB(Base):
    __tablename__ = "csv_files"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=now_ts)
    is_sample: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

