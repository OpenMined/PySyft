import pytest

from syftbox import __version__
from syftbox.client.core import SyftBoxRunner, run_migration
from syftbox.client.exceptions import SyftBoxAlreadyRunning
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.exceptions import SyftBoxException
from syftbox.lib.http import HEADER_SYFTBOX_VERSION


def test_client_single_instance(tmp_path):
    """Test that only one client instance can run"""
    config = SyftClientConfig(
        path=tmp_path / "config.json",
        data_dir=tmp_path,
        email="test@example.com",
        server_url="http://localhost:5001",
        client_url="http://localhost:8080",
    )
    client1 = SyftBoxRunner(config)
    client2 = SyftBoxRunner(config)

    client1.pid.create()

    # check should raise
    with pytest.raises(SyftBoxAlreadyRunning):
        client2.check_pidfile()

    # start should raise
    with pytest.raises(SyftBoxAlreadyRunning):
        client2.start()

    client1.shutdown()


def test_client_init_datasite(mock_config):
    client = SyftBoxRunner(mock_config)
    client.init_datasite()

    assert client.datasite.is_dir()
    assert client.public_dir.is_dir()


def test_register_user(mock_config, httpx_mock):
    httpx_mock.add_response(json={"token": "dummy-token"}, headers={HEADER_SYFTBOX_VERSION: __version__})
    client = SyftBoxRunner(mock_config)
    client.register_self()
    assert client.config.token == "dummy-token"


def test_register_user_error(mock_config, httpx_mock):
    httpx_mock.add_response(status_code=503)
    client = SyftBoxRunner(mock_config)
    with pytest.raises(SyftBoxException):
        client.register_self()

    assert client.config.token is None


def test_client_paths(tmp_path):
    """Test that client paths are correctly set up"""
    config = SyftClientConfig(
        path=tmp_path / "config.json",
        data_dir=tmp_path,
        email="test@example.com",
        server_url="http://localhost:5001",
        client_url="http://localhost:8080",
    )
    client = SyftBoxRunner(config)

    # data_dir should be the root of the client workspace
    assert client.workspace.data_dir == tmp_path


def test_migration(mock_config):
    # setup old datasites
    datasites = ["test@openmined.org", "test2@openmined.org"]
    old_dirs = ["apps", ".syft"] + datasites
    for dir in old_dirs:
        (mock_config.data_dir / dir).mkdir(parents=True)
    (mock_config.data_dir / ".syft" / "local_syncstate.json").touch()

    run_migration(mock_config)

    # check new workspace
    assert (mock_config.data_dir / "apis").is_dir()
    assert (mock_config.data_dir / "plugins").is_dir()
    assert (mock_config.data_dir / "datasites").is_dir()

    # check migrated datasites
    for ds in datasites:
        assert not (mock_config.data_dir / ds).exists()
        assert (mock_config.data_dir / "datasites" / ds).is_dir()

    # check syncstate migration
    assert not (mock_config.data_dir / ".syft").exists()
    assert (mock_config.data_dir / "plugins" / "local_syncstate.json").is_file()
