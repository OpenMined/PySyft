# stdlib
import logging
from typing import Dict
from typing import Generator

# third party
from _pytest.logging import LogCaptureFixture
from fastapi.testclient import TestClient
from loguru import logger
import pytest
from sqlalchemy.orm import Session

# grid absolute
from app.core.config import settings
from app.db.session import SessionLocal
from app.logger.handler import get_log_handler
from app.main import app
from app.tests.utils.user import authentication_token_from_email
from app.tests.utils.utils import get_superuser_token_headers

log_handler = get_log_handler()


@pytest.fixture(scope="session")
def db() -> Generator:
    yield SessionLocal()


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> Dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> Dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator:
    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    sink_handler_id = logger.add(PropagateHandler(), format=log_handler.format_record)
    yield caplog
    logger.remove(sink_handler_id)
