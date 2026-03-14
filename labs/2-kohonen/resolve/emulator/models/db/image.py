

from sqlalchemy import String, BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from base import Base, generate_uuid, now_ts

class ImageDB(Base):
    __tablename__ = "images"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=now_ts)