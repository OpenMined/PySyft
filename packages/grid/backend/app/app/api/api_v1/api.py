# third party
from fastapi import APIRouter

# grid absolute
from app.api.api_v1.endpoints import login
from app.api.api_v1.endpoints import syft
from app.api.api_v1.endpoints import users
from app.api.api_v1.endpoints import setup

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(syft.router, prefix="/syft", tags=["syfts"])
api_router.include_router(setup.router, prefix="/setup", tags=["setup"])
