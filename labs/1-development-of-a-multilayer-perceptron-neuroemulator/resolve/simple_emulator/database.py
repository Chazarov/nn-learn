from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config
from base import Base # type: ignore[unused-variable]

engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
