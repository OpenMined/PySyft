"""A manager builds one connection router and every part shares it, so the
whole manager has one connection, one transport and one peer store."""

from syft.sync.connections.base_connection import SyftboxPlatformConnection
from syft.sync.connections.drive.grdrive_config import GdriveConnectionConfig
from syft.sync.connections.drive.mock_drive_service import pair_with_mock_service
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig


def _config_with_both_roles() -> SyftboxManagerConfig:
    return SyftboxManagerConfig._base_config_for_testing(
        has_do_role=True, has_ds_role=True
    )


def test_every_part_of_manager_shares_one_router():
    manager = SyftboxManager.from_config(_config_with_both_roles())
    router = manager._connection_router

    assert manager.peer_manager.connection_router is router
    assert manager.datasite_owner_syncer.connection_router is router
    assert manager.datasite_watcher_syncer.connection_router is router
    watcher_cache = manager.datasite_watcher_syncer.datasite_watcher_cache
    assert watcher_cache.connection_router is router
    assert manager.peer_manager.peer_store is router.peer_store
    assert manager._peer_store is router.peer_store


def test_manager_builds_one_connection_from_its_config(monkeypatch):
    config = _config_with_both_roles()
    config.connection_configs = [
        GdriveConnectionConfig(email=config.email, token_path=None)
    ]
    built = []

    def build_mock_connection(connection_config):
        connection, _ = pair_with_mock_service(config.email, "peer@example.com")
        built.append(connection)
        return connection

    monkeypatch.setattr(
        SyftboxPlatformConnection, "from_config", staticmethod(build_mock_connection)
    )
    manager = SyftboxManager.from_config(config)

    assert manager._connection_router.connections == built
    assert len(built) == 1


def test_added_mock_connection_is_added_once_and_backs_the_transport():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    [connection] = do_manager._connection_router.connections

    assert do_manager._connection_router.transport.p2p._conn is connection
