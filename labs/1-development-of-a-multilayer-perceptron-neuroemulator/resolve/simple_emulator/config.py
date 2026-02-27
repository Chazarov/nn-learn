import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    JWT_SEKRET = os.getenv("JWT_SEKRET", "default-secret-change-me")
    JWT_EXPIRES_AT = int(os.getenv("JWT_EXPIRES_AT", "3600"))

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/perceptron_db")

    IMAGES_DIRECTORY = "data/visualisation"
    CSV_DIRECTORY = "data/learn"
    WEIGHTS_DIRECTORY = "data/weights"

config = Config()
