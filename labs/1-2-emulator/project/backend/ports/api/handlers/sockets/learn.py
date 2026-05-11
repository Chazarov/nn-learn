import asyncio

from fastapi import APIRouter, WebSocket, Query
import redis

from ports.celery.tasks import train_perceptron_task
from container import auth_service, project_service
from config import config
from models.ws_models import WSQueueUpdate, WSTrainingCompleted, WSError
from exceptions.auth_exception import AuthException
from exceptions.not_found import NotFoundException
from log import logger

router = APIRouter()

_redis = redis.Redis.from_url(config.REDIS_URL) #type: ignore

@router.websocket("/learn")
async def ws_learn(websocket: WebSocket, token: str = Query(...)):
    await websocket.accept()

    try:
        payload = auth_service.token_validate(token)
    except AuthException:
        await websocket.send_json(WSError(detail="Unauthorized").model_dump())
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
    except Exception:
        await websocket.send_json(WSError(detail="Invalid request").model_dump())
        await websocket.close()
        return

    required_fields = ["project_id", "activation_type", "epochs", "learning_rate"]
    if not all(f in data for f in required_fields):
        await websocket.send_json(
            WSError(detail=f"Missing required fields: {required_fields}").model_dump()
        )
        await websocket.close()
        return

    try:
        project_service.get_project(payload.user_id, data["project_id"])
    except NotFoundException:
        await websocket.send_json(WSError(detail="Project not found").model_dump())
        await websocket.close()
        return

    task = train_perceptron_task.delay( #type: ignore
        user_id=payload.user_id,
        project_id=data["project_id"],
        activation_type=data["activation_type"],
        softmax_use=data.get("softmax_use", False),
        loss_type=data.get("loss_type", "MSE"),
        epochs=data["epochs"],
        learning_rate=data["learning_rate"],
    )

    try:
        while True:
            state = task.state

            if state == "PENDING":
                position:int = _redis.llen(config.CELERY_QUEUE_KEY) #type: ignore
                
                await websocket.send_json(
                    WSQueueUpdate(position=position).model_dump()
                )

            elif state == "SUCCESS":
                result = task.result
                await websocket.send_json(
                    WSTrainingCompleted(
                        epochs=result["epochs"],
                        loss=result["loss"],
                        project=result["project"],
                        image_id=result["image_id"],
                    ).model_dump()
                )
                break

            elif state == "FAILURE":
                await websocket.send_json(
                    WSError(detail="Training failed").model_dump()
                )
                break

            await asyncio.sleep(config.POLL_INTERVAL_SECONDS)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
