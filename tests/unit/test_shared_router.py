"""A manager builds one connection router and every part shares it, so the
whole manager has one transport, one connection and one peer store."""

import pytest

from syft.sync.connections.base_connection import SyftboxPlatformConnection
from syft.sync.connections.drive.drive_connections import drive_connection
from syft.sync.connections.drive.grdrive_config import GdriveConnectionConfig
from syft.sync.connections.drive.mock_drive_service import pair_with_mock_service
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig


def _config_with_both_roles() -> SyftboxManagerConfig:
    return SyftboxManagerConfig._base_config_for_testing(
        has_do_role=True, has_ds_role=True
    )


def _mock_connection(email: str):
    connection, _ = pair_with_mock_service(email, "peer@example.com")
    return connection


def test_every_part_of_manager_shares_one_router():
    config = _config_with_both_roles()
    transport = _mock_connection(config.email).as_transport()
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


def test_manager_builds_one_connection_from_its_config(monkeypatch):
    config = _config_with_both_roles()
    config.connection_config = GdriveConnectionConfig(
        email=config.email, token_path=None
    )
    built = []

    def build_mock_connection(connection_config):
        built.append(_mock_connection(config.email))
        return built[-1]

    monkeypatch.setattr(
        SyftboxPlatformConnection, "from_config", staticmethod(build_mock_connection)
    )
    manager = SyftboxManager.from_config(config)

    assert len(built) == 1
    assert drive_connection(manager._connection_router.transport) is built[0]


def test_every_transport_part_shares_one_mock_connection():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    transport = do_manager._connection_router.transport
    connection = drive_connection(transport)

    for part in (
        transport.p2p,
        transport.owner_store,
        transport.watcher_store,
        transport.collections,
        transport.peering,
    ):
        assert part._conn is connection


def test_manager_without_backend_fails_at_init():
    with pytest.raises(ValueError, match="names no backend"):
        SyftboxManager.from_config(_config_with_both_roles())
