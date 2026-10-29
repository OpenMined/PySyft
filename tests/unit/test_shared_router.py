"""A manager builds one connection router and every part shares it, so the
whole manager has one transport, one connection and one peer store."""

import pytest

from syft.sync.connections.drive.drive_connections import drive_connection
from syft.sync.connections.drive.gdrive_transport import GDriveConnection
from syft.sync.connections.drive.grdrive_config import GdriveConnectionConfig
from syft.sync.connections.drive.mock_drive_service import pair_with_mock_service
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft.sync.transport.config import TransportConfig


def _config_with_both_roles() -> SyftboxManagerConfig:
    return SyftboxManagerConfig._base_config_for_testing(
        has_do_role=True, has_ds_role=True
    )


def _mock_connection(email: str) -> GDriveConnection:
    connection, _ = pair_with_mock_service(email, "peer@example.com")
    return connection


def test_every_part_of_manager_shares_one_router():
    config = _config_with_both_roles()
    transport = SyftboxManager._drive_transport(_mock_connection(config.email))
    manager = SyftboxManager.from_config(config, transport)
    router = manager._connection_router

    assert router.transport is transport
    assert manager.peer_manager.connection_router is router
    assert manager.datasite_owner_syncer.connection_router is router
    assert manager.datasite_watcher_syncer.connection_router is router
    watcher_cache = manager.datasite_watcher_syncer.datasite_watcher_cache
    assert watcher_cache.connection_router is router
    assert manager.peer_manager.peer_store is router.peer_store
    assert manager._peer_store is router.peer_store


def test_manager_builds_one_drive_connection_from_its_config(monkeypatch):
    built = []

    def build_mock_connection(connection_config):
        built.append(_mock_connection(connection_config.email))
        return built[-1]

    monkeypatch.setattr(GDriveConnection, "from_config", build_mock_connection)
    config = _config_with_both_roles()
    config.connection_config = GdriveConnectionConfig(
        email=config.email, token_path=None
    )
    manager = SyftboxManager.from_config(config)

    [connection] = built
    for component in manager._connection_router.transport.components():
        assert component._conn is connection


def test_every_transport_component_shares_one_mock_connection():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    transport = do_manager._connection_router.transport
    connection = drive_connection(transport)

    for component in transport.components():
        assert component._conn is connection


def test_manager_without_connection_config_fails_at_init():
    with pytest.raises(ValueError, match="names no connection"):
        SyftboxManager.from_config(_config_with_both_roles())


def test_component_other_than_drive_is_not_built_yet():
    config = _config_with_both_roles()
    config.connection_config = GdriveConnectionConfig(
        email=config.email, token_path=None
    )
    config.transport_config = TransportConfig(p2p="webrtc")

    with pytest.raises(NotImplementedError, match="Only Google Drive"):
        SyftboxManager.from_config(config)
