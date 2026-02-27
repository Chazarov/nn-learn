from fastapi import APIRouter

from api.handlers.perceptrone_actions import router as p_actions_router
from api.handlers.images import router as images_router
from api.handlers.csv_files import router as csv_router
from api.handlers.auth import router as auth_router

main_router = APIRouter(prefix="/api")

main_router.include_router(auth_router, prefix="/auth", tags=["auth"])
main_router.include_router(p_actions_router, prefix="/actions", tags=["actions"])
main_router.include_router(images_router, prefix="/images", tags=["images"])
main_router.include_router(csv_router, prefix="/csv", tags=["csv"])
