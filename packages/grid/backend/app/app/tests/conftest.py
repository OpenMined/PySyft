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
from app.core.config import settings
from app.db.session import get_db_session
from app.initial_data import init_db
from app.logger.handler import get_log_handler

log_handler = get_log_handler()


@pytest.fixture(scope="session")
def db() -> Generator:
    yield get_db_session()


@pytest.fixture
async def app() -> FastAPI:
    session = get_db_session()
    init_db(db=session)

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

    sink_handler_id = logger.add(PropagateHandler(), format=log_handler.format_record)
    yield caplog
    logger.remove(sink_handler_id)
