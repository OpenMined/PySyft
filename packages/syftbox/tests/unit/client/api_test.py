import pytest
from fastapi.testclient import TestClient

from syftbox.client.api import create_api
from syftbox.client.base import SyftBoxContextInterface
from syftbox.lib.client_config import SyftClientConfig


class MockClient(SyftBoxContextInterface):
    def __init__(self):
        self.config = SyftClientConfig(
            path="/tmp/syftbox/config.yaml",
            client_url="http://localhost:8080",
            server_url="http://localhost:5000",
            email="test@user.com",
        )

    @property
    def all_datasites(self):
        return ["datasite1", "datasite2"]


@pytest.fixture
def mock_api():
    yield TestClient(create_api(MockClient()))


def test_create_api(mock_api):
    response = mock_api.get("/")
    assert response.status_code == 200


def test_version(mock_api):
    response = mock_api.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_datasites():
    app = create_api(MockClient())
    mock_api = TestClient(app)
    response = mock_api.get("/datasites")
    assert response.status_code == 200
    assert response.json() == {"datasites": ["datasite1", "datasite2"]}
