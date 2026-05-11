import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    JWT_SEKRET = os.getenv("JWT_SEKRET")
    JWT_EXPIRES_AT = 999999999

    DATABASE_URL = os.getenv("DATABASE_URL")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    IMAGES_DIRECTORY = "data/visualisation"
    CSV_DIRECTORY = "data/learn"
    WEIGHTS_DIRECTORY = "data/weights"
    SAMPLES_DIRECTORY = "samples"


    #Websockets

    CELERY_QUEUE_KEY = "celery"
    POLL_INTERVAL_SECONDS = 1.0

config = Config()
