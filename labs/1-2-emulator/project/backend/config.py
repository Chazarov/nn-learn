import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()


class NumConstraint(BaseModel):
    MAX_VALUE: None|int
    MIN_VALUE: None|int
    AWALIBLE_SIZES: None|List[int]
 


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




    class PublicConstraints:


        # Api constraints:

        ## Kohonen constraints:
        KOHONEN_INPUT_LAYER_SIZE = NumConstraint(MAX_VALUE=None, MIN_VALUE=None, AWALIBLE_SIZES=[4, 9, 16, 25, 36, 49])

        ## Perceptrone constraints:

        # TODO: Add constraints to all perceptron API constants that may cause perfomance issues.




config = Config()
