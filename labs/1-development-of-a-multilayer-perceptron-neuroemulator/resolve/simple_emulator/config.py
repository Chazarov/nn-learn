import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    JWT_SEKRET = os.getenv("JWT_SEKRET")
    JWT_EXPIRES_AT = os.getenv("JWT_EXPIRES_AT")

config = Config()
