from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import config

engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
