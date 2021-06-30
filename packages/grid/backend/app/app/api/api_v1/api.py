# third party
from fastapi import APIRouter

# grid absolute
from app.api.api_v1.endpoints import items
from app.api.api_v1.endpoints import login
from app.api.api_v1.endpoints import node
from app.api.api_v1.endpoints import users
from app.api.api_v1.endpoints import utils

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
# api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
# api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(node.router, prefix="/syft", tags=["syfts"])
