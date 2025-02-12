from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from syftbox import __version__
from syftbox.client.base import PluginManagerInterface, SyftBoxContextInterface
from syftbox.client.core import SyftBoxContext
from syftbox.client.server_client import SyftBoxClient
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.datasite import create_datasite
from syftbox.lib.http import HEADER_SYFTBOX_VERSION
from syftbox.lib.workspace import SyftWorkspace
from syftbox.server.migrations import run_migrations
from syftbox.server.server import create_server
from syftbox.server.settings import ServerSettings
from tests.unit.server.conftest import get_access_token


def authenticate_testclient(client: TestClient, email: str) -> None:
    access_token = get_access_token(client, email)
    client.headers["email"] = email
    client.headers["Authorization"] = f"Bearer {access_token}"
    client.headers[HEADER_SYFTBOX_VERSION] = __version__


class MockPluginManager(PluginManagerInterface):
    pass


def setup_datasite(tmp_path: Path, server_client: TestClient, email: str) -> SyftBoxContextInterface:
    data_dir = tmp_path / email
    config = SyftClientConfig(
        path=data_dir / "config.json",
        data_dir=data_dir,
        email=email,
        server_url=str(server_client.base_url),
        client_url="http://localhost:8080",
    )
    config.save()
    ws = SyftWorkspace(config.data_dir)
    ws.mkdirs()
    context = SyftBoxContext(
        config,
        ws,
        SyftBoxClient(conn=server_client),
        MockPluginManager(),
    )
    create_datasite(context)
    authenticate_testclient(server_client, email)

    return context


@pytest.fixture(scope="function")
def server_app(tmp_path: Path) -> FastAPI:
    """
    NOTE we are spawning a new server thread for each datasite,
    this is not ideal but it is the same as using multiple uvicorn workers
    """
    path = tmp_path / "server"
    path.mkdir()
    settings = ServerSettings.from_data_folder(path)
    settings.auth_enabled = False
    settings.otel_enabled = False
    server_app = create_server(settings)
    run_migrations(settings)
    return server_app


@pytest.fixture()
def datasite_1(tmp_path: Path, server_app: FastAPI) -> SyftBoxContextInterface:
    email = "user_1@openmined.org"
    with TestClient(server_app) as client:
        client.headers[HEADER_SYFTBOX_VERSION] = __version__
        return setup_datasite(tmp_path, client, email)


@pytest.fixture()
def datasite_2(tmp_path: Path, server_app: FastAPI) -> SyftBoxContextInterface:
    email = "user_2@openmined.org"
    with TestClient(server_app) as client:
        client.headers[HEADER_SYFTBOX_VERSION] = __version__
        return setup_datasite(tmp_path, client, email)


@pytest.fixture(scope="function")
def server_client(server_app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(server_app) as client:
        client.headers[HEADER_SYFTBOX_VERSION] = __version__
        yield client
