import math
import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import AfterValidator, BaseModel

load_dotenv()


class NumConstraint(BaseModel):
    """If ``allowed_values`` is set, only those ints are accepted; otherwise ``min_value``..``max_value``."""

    min_value: int = 0
    max_value: int = 100_000
    allowed_values: Optional[List[int]] = None


class FloatConstraint(BaseModel):
    """Inclusive float range; non-finite values (NaN/inf) are rejected."""

    min_value: float
    max_value: float
    

class HiddenLayersConstraint(BaseModel):
    """Bounds ``hidden_layers_architecture`` (list length and each hidden layer width)."""

    max_hidden_layers: int
    neurons_per_hidden_layer: NumConstraint


def num_constraint_validator(constraint: NumConstraint) -> AfterValidator:
    """``AfterValidator`` for ``Annotated[int, ...]`` on FastAPI ``Body`` parameters."""
    allowed = constraint.allowed_values
    lo, hi = constraint.min_value, constraint.max_value
    allowed_set = frozenset(allowed) if allowed is not None else None

    def _validate(n: int) -> int:
        if allowed_set is not None:
            if n not in allowed_set:
                raise ValueError(f"only values from {sorted(allowed_set)} are allowed")
            return n
        if n < lo or n > hi:
            raise ValueError(f"expected {lo} <= value <= {hi}")
        return n

    return AfterValidator(_validate)


def float_constraint_validator(constraint: FloatConstraint) -> AfterValidator:
    """``AfterValidator`` for ``Annotated[float, ...]`` on FastAPI ``Body`` parameters."""

    lo, hi = constraint.min_value, constraint.max_value

    def _validate(x: float) -> float:
        if not math.isfinite(x):
            raise ValueError("value must be a finite number")
        if x < lo or x > hi:
            raise ValueError(f"expected {lo} <= value <= {hi}")
        return x

    return AfterValidator(_validate)


def hidden_layers_list_validator(constraint: HiddenLayersConstraint) -> AfterValidator:
    """``AfterValidator`` for ``Annotated[list[int], ...]`` (perceptron hidden layer sizes)."""
    c = constraint.neurons_per_hidden_layer
    allowed_set = frozenset(c.allowed_values) if c.allowed_values is not None else None
    lo, hi = c.min_value, c.max_value
    max_n = constraint.max_hidden_layers

    def _validate(layers: List[int]) -> List[int]:
        if len(layers) > max_n:
            raise ValueError(f"at most {max_n} hidden layers are allowed")
        for idx, n in enumerate(layers):
            if allowed_set is not None:
                if n not in allowed_set:
                    raise ValueError(
                        f"hidden layer {idx}: only values from {sorted(allowed_set)} are allowed"
                    )
            elif n < lo or n > hi:
                raise ValueError(f"hidden layer {idx}: expected {lo} <= neurons <= {hi}")
        return layers

    return AfterValidator(_validate)


def finite_input_vector_validator(max_elements: int) -> AfterValidator:
    """Reject oversized lists and non-finite coordinates (DoS / stable numpy)."""

    def _validate(v: List[float]) -> List[float]:
        if len(v) > max_elements:
            raise ValueError(f"input_vector must have at most {max_elements} elements")
        for i, x in enumerate(v):
            xf = float(x)
            if not math.isfinite(xf):
                raise ValueError(f"non-finite value at index {i}")
        return v

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
        # --- Kohonen (API + training cost) ---
        KOHONEN_INPUT_FEATURES_MAX = 20

        KOHONEN_OUTPUT_LAYER_SIZE_RANGE = NumConstraint(
            allowed_values=[4, 9, 16, 25, 36, 49],
        )

        KOHONEN_INPUT_LAYER_SIZE_RANGE = NumConstraint(
            min_value=1,
            max_value=KOHONEN_INPUT_FEATURES_MAX,
        )

        KOHONEN_LEARN_EPOCHS_RANGE = NumConstraint(
            min_value=1,
            max_value=100_000,
        )

        KOHONEN_LEARN_LEARNING_RATE_RANGE = FloatConstraint(
            min_value=1e-8,
            max_value=1.0,
        )

        KOHONEN_INITIAL_NEIGHBORHOOD_RADIUS_RANGE = FloatConstraint(
            min_value=1e-8,
            max_value=1_000.0,
        )

        KOHONEN_GET_ANSWER_INPUT_VECTOR_MAX_LEN = KOHONEN_INPUT_FEATURES_MAX

        # --- Perceptron (architecture size / train loop cost) ---
        PERCEPTRON_HIDDEN_LAYERS = HiddenLayersConstraint(
            max_hidden_layers=12,
            neurons_per_hidden_layer=NumConstraint(min_value=1, max_value=512),
        )

        PERCEPTRON_LEARN_EPOCHS_RANGE = NumConstraint(
            min_value=1,
            max_value=100_000,
        )

        PERCEPTRON_LEARN_LEARNING_RATE_RANGE = FloatConstraint(
            min_value=1e-8,
            max_value=1.0,
        )

        # Raise if your CSV feature count can exceed this (caps ``get_answer`` payload size).
        PERCEPTRON_GET_ANSWER_INPUT_VECTOR_MAX_LEN = 512




config = Config()
