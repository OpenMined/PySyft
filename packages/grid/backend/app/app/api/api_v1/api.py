# third party
from fastapi import APIRouter

# grid absolute
from app.api.api_v1.endpoints import login, setup, syft, users

api_router = APIRouter()
api_router.include_router(login.router, tags=["Grid Login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(syft.router, prefix="/syft", tags=["syfts"])
api_router.include_router(setup.router, prefix="/setup", tags=["setup"])
