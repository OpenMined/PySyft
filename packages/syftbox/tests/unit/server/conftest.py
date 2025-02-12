import json

import pytest
from fastapi.testclient import TestClient

from syftbox import __version__
from syftbox.client.server_client import SyncClient
from syftbox.lib.constants import PERM_FILE
from syftbox.lib.http import HEADER_SYFTBOX_VERSION
from syftbox.server.migrations import run_migrations
from syftbox.server.server import create_server
from syftbox.server.settings import ServerSettings

TEST_DATASITE_NAME = "test_datasite@openmined.org"
TEST_FILE = "test_file.txt"
PERMFILE_DICT = [
    {
        "path": "*",
        "user": "*",
        "permissions": ["admin", "read", "write"],
    },
    {
        "path": "**/*",
        "user": "*",
        "permissions": ["admin", "read", "write"],
    },
]


def get_access_token(client: TestClient, email: str) -> str:
    response = client.post("/auth/request_email_token", json={"email": email})
    email_token = response.json()["email_token"]
    response = client.post(
        "/auth/validate_email_token",
        headers={"Authorization": f"Bearer {email_token}"},
        params={"email": email},
    )

    if response.status_code != 200:
        raise ValueError(f"Failed to get access token, {response.text}")
    return response.json()["access_token"]


@pytest.fixture(scope="function")
def client(monkeypatch, tmp_path):
    """Every client gets their own snapshot folder at `tmp_path`"""
    snapshot_folder = tmp_path / "snapshot"
    settings = ServerSettings.from_data_folder(snapshot_folder)
    settings.auth_enabled = False
    monkeypatch.setenv("SYFTBOX_DATA_FOLDER", str(settings.data_folder))
    monkeypatch.setenv("SYFTBOX_SNAPSHOT_FOLDER", str(settings.snapshot_folder))
    monkeypatch.setenv("SYFTBOX_USER_FILE_PATH", str(settings.user_file_path))
    monkeypatch.setenv("SYFTBOX_OTEL_ENABLED", str(False))
    monkeypatch.setenv("SYFTBOX_AUTH_ENABLED", str(False))

    datasite_name = TEST_DATASITE_NAME
    datasite = settings.snapshot_folder / datasite_name
    datasite.mkdir(parents=True)

    datafile = datasite / TEST_FILE
    datafile.touch()
    datafile.write_bytes(b"Hello, World!")

    datafile = datasite / TEST_DATASITE_NAME / TEST_FILE
    datafile.parent.mkdir(parents=True)

    datafile.touch()
    datafile.write_bytes(b"Hello, World!")

    permfile = datasite / PERM_FILE
    permfile.touch()
    permfile.write_text(json.dumps(PERMFILE_DICT))

    server_app = create_server(settings)
    run_migrations(settings)
    with TestClient(server_app) as client:
        client.headers[HEADER_SYFTBOX_VERSION] = __version__
        access_token = get_access_token(client, TEST_DATASITE_NAME)
        client.headers["Authorization"] = f"Bearer {access_token}"
        yield client


@pytest.fixture(scope="function")
def sync_client(client: TestClient):
    return SyncClient(conn=client)


@pytest.fixture(scope="function")
def client_without_perms(monkeypatch, tmp_path):
    """Every client gets their own snapshot folder at `tmp_path`"""
    settings = ServerSettings.from_data_folder(tmp_path)
    settings.otel_enabled = False
    settings.auth_enabled = False

    monkeypatch.setenv("SYFTBOX_DATA_FOLDER", str(settings.data_folder))
    monkeypatch.setenv("SYFTBOX_SNAPSHOT_FOLDER", str(settings.snapshot_folder))
    monkeypatch.setenv("SYFTBOX_USER_FILE_PATH", str(settings.user_file_path))
    monkeypatch.setenv("SYFTBOX_OTEL_ENABLED", str(False))
    monkeypatch.setenv("SYFTBOX_AUTH_ENABLED", str(False))

    datasite_name = TEST_DATASITE_NAME
    datasite = settings.snapshot_folder / datasite_name
    datasite.mkdir(parents=True)

    datafile = datasite / TEST_FILE
    datafile.touch()
    datafile.write_bytes(b"Hello, World!")

    permfile = datasite / PERM_FILE
    permfile.touch()
    permfile.write_text("")

    server_app = create_server(settings)
    run_migrations(settings)
    with TestClient(server_app) as client:
        client.headers[HEADER_SYFTBOX_VERSION] = __version__
        access_token = get_access_token(client, TEST_DATASITE_NAME)
        client.headers["Authorization"] = f"Bearer {access_token}"
        yield client
