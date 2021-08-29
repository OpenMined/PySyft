# stdlib
import logging
from typing import Generator

# third party
from _pytest.logging import LogCaptureFixture
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import AsyncClient
from loguru import logger
import pytest

# grid absolute
from app.db.session import SessionLocal

# from app.main import app
# from app.tests.utils.utils import get_superuser_token_headers

from app.logger.handler import get_log_handler

log_handler = get_log_handler()


@pytest.fixture(scope="session")
def db() -> Generator:
    yield SessionLocal()


@pytest.fixture
async def app() -> FastAPI:
    # grid absolute
    from app.main import app

    async with LifespanManager(app):
        yield app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    async with AsyncClient(
        app=app,
        base_url=settings.SERVER_HOST,
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator:
    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

<<<<<<< HEAD
# @pytest.fixture(scope="module")
# def normal_user_token_headers(client: TestClient, db: Session) -> Dict[str, str]:
#     return authentication_token_from_email(
#         client=client, email=settings.EMAIL_TEST_USER, db=db
#     )
=======
    sink_handler_id = logger.add(PropagateHandler(), format=log_handler.format_record)
    yield caplog
    logger.remove(sink_handler_id)
>>>>>>> cee4ed6ea3a82659d407835df260dce05b6a2c06
