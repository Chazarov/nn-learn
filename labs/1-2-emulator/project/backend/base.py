import time
import uuid

from sqlalchemy.orm import DeclarativeBase



def generate_uuid() -> str: #type: ignore
    return str(uuid.uuid4())

def now_ts() -> int: #type: ignore
    return int(time.time())

class Base(DeclarativeBase):
    pass
