import uvicorn
from fastapi import FastAPI

from database import engine, Base
import models.db_models # type: ignore # noqa: F401 — регистрация таблиц в Base.metadata
from celery_app import celery_app as _celery_app  # type: ignore # noqa: F401 — set configured app as current
from ports.api.routes import main_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multilayer Perceptron API")
app.include_router(main_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
