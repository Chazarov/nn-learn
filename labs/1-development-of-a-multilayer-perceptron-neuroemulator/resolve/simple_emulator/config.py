import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    JWT_SEKRET = os.getenv("JWT_SEKRET")
    JWT_EXPIRES_AT = 3600

    DATABASE_URL = os.getenv("DATABASE_URL")

    IMAGES_DIRECTORY = "data/visualisation"
    CSV_DIRECTORY = "data/learn"
    WEIGHTS_DIRECTORY = "data/weights"

config = Config()
