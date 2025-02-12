from pathlib import Path

import pytest

from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.client_shim import Client


@pytest.fixture
def client(mock_config):
    return Client(conf=mock_config)


def test_client_properties(client, mock_config):
    assert client.email == mock_config.email
    assert client.config_path == mock_config.path
    assert client.my_datasite == client.workspace.datasites / mock_config.email
    assert client.datasites == client.workspace.datasites
    assert client.sync_folder == client.workspace.datasites  # Test deprecated property
    assert client.datasite_path == client.workspace.datasites / mock_config.email  # Test deprecated property


def test_client_load(mock_config, tmp_path):
    _ = mock_config  # this fixture creates and saves a mock tmp_path/config.json file
    config_path = tmp_path / "config.json"
    client = Client.load(config_path)
    assert isinstance(client, Client)
    assert isinstance(client.config, SyftClientConfig)


def test_api_request_name(client, monkeypatch):
    test_dir = Path("/fake/api/request/path")
    monkeypatch.setattr(Path, "cwd", lambda: test_dir)
    assert client.api_request_name == "path"


@pytest.mark.parametrize(
    "api_name,datasite,expected",
    [
        ("test_api", None, "test_api"),
        (None, "other@example.com", "current_api"),
        ("custom_api", "other@example.com", "custom_api"),
    ],
)
def test_api_data(client, monkeypatch, api_name, datasite, expected):
    monkeypatch.setattr(Path, "cwd", lambda: Path("/fake/current_api"))

    result = client.api_data(api_name, datasite)

    datasite_email = datasite or client.config.email
    expected_path = client.workspace.datasites / datasite_email / "api_data" / (expected)
    assert result == expected_path


def test_makedirs(client, tmp_path):
    test_paths = [tmp_path / "dir1", tmp_path / "dir2" / "subdir", tmp_path / "dir3" / "subdir" / "subsubdir"]

    client.makedirs(*test_paths)

    for path in test_paths:
        assert path.exists()
        assert path.is_dir()


def test_makedirs_existing(client, tmp_path):
    test_path = tmp_path / "existing"
    test_path.mkdir()

    # Should not raise error when directory exists
    client.makedirs(test_path)
    assert test_path.exists()
