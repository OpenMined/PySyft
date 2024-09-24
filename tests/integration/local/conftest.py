# stdlib
from collections.abc import Generator
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from secrets import token_hex
from typing import Any

# third party
import pytest
import pytest_asyncio

# syft absolute
import syft as sy
from syft.orchestra import ClientAlias
from syft.orchestra import ServerHandle
from syft.util.test_helpers.email_helpers import EmailServer
from syft.util.test_helpers.email_helpers import SMTPTestServer
from syft.util.test_helpers.email_helpers import get_email_server


@pytest.fixture()
def server_args() -> dict[str, Any]:
    return {}


@pytest.fixture
def server(server_args: dict[str, Any]) -> Generator[ServerHandle, None, None]:
    _server = sy.orchestra.launch(
        **{
            "name": token_hex(8),
            "dev_mode": False,
            "reset": True,
            "queue_port": None,
            **server_args,
        }
    )
    # startup code here
    yield _server
    # Cleanup code
    if (python_server := _server.python_server) is not None:
        python_server.cleanup()
    _server.land()


@pytest.fixture
def client(server: ServerHandle) -> ClientAlias:
    return server.login(email="info@openmined.org", password="changethis")


@pytest_asyncio.fixture(scope="function")
async def email_wrapper() -> Any:
    email_server, smtp_server = get_email_server()

    @dataclass
    class EmailServerWrapper:
        email_server: EmailServer
        smtp_server: SMTPTestServer

    email_wrapper = EmailServerWrapper(
        email_server=email_server, smtp_server=smtp_server
    )
    yield email_wrapper

    # Finish SMTP process.
    smtp_server.stop()


def matrix(
    *,
    excludes_: Iterable[dict[str, Any]] | None = None,
    **kwargs: Iterable,
) -> list[dict[str, Any]]:
    args = ([(k, v) for v in vs] for k, vs in kwargs.items())
    args = product(*args)

    excludes_ = [] if excludes_ is None else [kv.items() for kv in excludes_]

    args = (
        arg
        for arg in args
        if not any(all(kv in arg for kv in kvs) for kvs in excludes_)
    )

    return [dict(kvs) for kvs in args]
