# third party
from fastapi import APIRouter

# grid absolute
from grid.api.association_requests import association_requests
from grid.api.auth import login
from grid.api.auth import register
from grid.api.datasets import datasets
from grid.api.meta import ping
from grid.api.meta import status
from grid.api.requests.routes import router as requests_router
from grid.api.roles import roles
from grid.api.setup import setup
from grid.api.syft import syft
from grid.api.users.routes import router as user_router

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(register.router, tags=["register"])
api_router.include_router(user_router, prefix="/users", tags=["users"])
api_router.include_router(requests_router, prefix="/requests", tags=["requests"])
api_router.include_router(roles.router, prefix="/roles", tags=["roles"])
api_router.include_router(syft.router, prefix="/syft", tags=["syft"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(setup.router, prefix="/setup", tags=["setup"])
api_router.include_router(
    association_requests.router,
    prefix="/association-requests",
    tags=["association-requests"],
)
api_router.include_router(status.router, prefix="/status")
api_router.include_router(ping.router, prefix="/ping")
