from celery import Celery

from config import config

celery_app = Celery(
    "simple_emulator",
    broker=config.REDIS_URL,
    backend=config.REDIS_URL,
)

celery_app.autodiscover_tasks(["ports.celery"])
