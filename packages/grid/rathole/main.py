# stdlib
from enum import Enum
import os
import sys

# third party
from fastapi import FastAPI
from fastapi import status
from loguru import logger

# relative
from .models import RatholeConfig
from .models import ResponseModel
from .utils import RatholeClientToml
from .utils import RatholeServerToml

# Logging Configuration
log_level = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, colorize=True, level=log_level)

app = FastAPI(title="Rathole")


class RatholeMode(Enum):
    CLIENT = "client"
    SERVER = "server"


ServiceType = os.getenv("RATHOLE_MODE", "client").lower()


RatholeTomlManager = (
    RatholeServerToml()
    if ServiceType == RatholeMode.SERVER.value
    else RatholeClientToml()
)


async def healthcheck() -> bool:
    return True


@app.get(
    "/healthcheck",
    response_model=ResponseModel,
    status_code=status.HTTP_200_OK,
)
async def healthcheck_endpoint() -> ResponseModel:
    res = await healthcheck()
    if res:
        return ResponseModel(message="OK")
    else:
        return ResponseModel(message="FAIL")


@app.post(
    "/config/",
    response_model=ResponseModel,
    status_code=status.HTTP_201_CREATED,
)
async def add_config(config: RatholeConfig) -> ResponseModel:
    RatholeTomlManager.add_config(config)
    return ResponseModel(message="Config added successfully")


@app.delete(
    "/config/{uuid}",
    response_model=ResponseModel,
    status_code=status.HTTP_200_OK,
)
async def remove_config(uuid: str) -> ResponseModel:
    RatholeTomlManager.remove_config(uuid)
    return ResponseModel(message="Config removed successfully")


@app.put(
    "/config/{uuid}",
    response_model=ResponseModel,
    status_code=status.HTTP_200_OK,
)
async def update_config(config: RatholeConfig) -> ResponseModel:
    RatholeTomlManager.update_config(config=config)
    return ResponseModel(message="Config updated successfully")


@app.get(
    "/config/{uuid}",
    response_model=RatholeConfig,
    status_code=status.HTTP_201_CREATED,
)
async def get_config(uuid: str) -> RatholeConfig:
    return RatholeTomlManager.get_config(uuid)
