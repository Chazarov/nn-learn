from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class WSMessageType(str, Enum):
    QUEUE_UPDATE = "queue_update"
    TRAINING_COMPLETED = "training_completed"
    ERROR = "error"


class WSQueueUpdate(BaseModel):
    type: WSMessageType = WSMessageType.QUEUE_UPDATE
    position: int


class WSTrainingCompleted(BaseModel):
    type: WSMessageType = WSMessageType.TRAINING_COMPLETED
    epochs: int
    loss: float
    project: Dict[str, Any]
    image_id: str


class WSError(BaseModel):
    type: WSMessageType = WSMessageType.ERROR
    detail: str
