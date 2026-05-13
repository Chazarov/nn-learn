import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import AfterValidator, BaseModel

load_dotenv()


class NumConstraint(BaseModel):
    """Если allowed_values не None — допускаются только эти числа; иначе проверка по min_value..max_value."""

    min_value: int = 0
    max_value: int = 100_000
    allowed_values: Optional[List[int]] = None


def num_constraint_validator(constraint: NumConstraint) -> AfterValidator:
    """Возвращает AfterValidator для Annotated[int, ...] = Body(...) в хендлерах FastAPI."""
    allowed = constraint.allowed_values
    lo, hi = constraint.min_value, constraint.max_value
    allowed_set = frozenset(allowed) if allowed is not None else None

    def _validate(n: int) -> int:
        if allowed_set is not None:
            if n not in allowed_set:
                raise ValueError(f"допустимы только значения из {sorted(allowed_set)}")
            return n
        if n < lo or n > hi:
            raise ValueError(f"ожидается {lo} <= значение <= {hi}")
        return n

    return AfterValidator(_validate)
 


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
        KOHONEN_OUTPUT_LAYER_SIZE_RANGE = NumConstraint(
            name="KOHONEN_OUTPUT_LAYER_SIZE_RANGE",
            allowed_values=[4, 9, 16, 25, 36, 49],
        )

        KOHONEN_INPUT_LAYER_SIZE_RANGE = NumConstraint(
            name="KOHONEN_INPUT_LAYER_SIZE_RANGE",
            min_value=1,
            max_value=20,
        )

        ## Perceptrone constraints:

        # TODO: Add constraints to all perceptron API constants that may cause perfomance issues.




config = Config()
