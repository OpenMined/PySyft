# third party
from fastapi import APIRouter

# grid absolute
from app.api.api_v1.endpoints import association_requests
from app.api.api_v1.endpoints import datasets
from app.api.api_v1.endpoints import login
from app.api.api_v1.endpoints import roles
from app.api.api_v1.endpoints import setup
from app.api.api_v1.endpoints import syft
from app.api.api_v1.endpoints import users

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["user"])
api_router.include_router(roles.router, prefix="/roles", tags=["roles"])
api_router.include_router(syft.router, prefix="/syft", tags=["syft"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(setup.router, prefix="/setup", tags=["setup"])
api_router.include_router(
    association_requests.router,
    prefix="/association-requests",
    tags=["association-requests"],
)
