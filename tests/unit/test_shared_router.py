"""A manager builds one connection router and every part shares it, so the
whole manager has one transport, one connection and one peer store."""

from unittest.mock import Mock

import pytest

from syft.sync.connections.base_connection import ConnectionConfig
from syft.sync.connections.drive.drive_connections import (
    DriveWatcherStore,
    drive_cached_client_of,
)
from syft.sync.connections.drive.drive_cached_client import DriveCachedClient
from syft.sync.connections.drive.grdrive_config import GdriveConnectionConfig
from syft.sync.connections.drive.mock_drive_service import pair_with_mock_service
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft.sync.transport.config import TransportConfig
from syft.sync.transport.protocols import OwnerStore


def _config_with_both_roles() -> SyftboxManagerConfig:
    return SyftboxManagerConfig._base_config_for_testing(
        has_do_role=True, has_ds_role=True
    )


def _drive_config(email: str) -> GdriveConnectionConfig:
    return GdriveConnectionConfig(email=email, token_path=None)


def _mock_cached_client(email: str) -> DriveCachedClient:
    cached_client, _ = pair_with_mock_service(email, "peer@example.com")
    return cached_client


def test_every_part_of_manager_shares_one_router():
    config = _config_with_both_roles()
    transport = SyftboxManager._drive_transport(_mock_cached_client(config.email))
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


def _drive_cached_clients(transport) -> list[DriveCachedClient]:
    """The cached client of each component that uses Drive. The watcher store does not
    use it yet."""
    return [
        component._cached_client
        for component in transport.components()
        if not isinstance(component, DriveWatcherStore)
    ]


def test_manager_builds_one_drive_connection_from_its_config(monkeypatch):
    built = []

    def build_mock_cached_client(connection_config):
        built.append(_mock_cached_client(connection_config.email))
        return built[-1]

    monkeypatch.setattr(DriveCachedClient, "from_config", build_mock_cached_client)
    config = _config_with_both_roles()
    config.transport_config = TransportConfig.one_connection(
        _drive_config(config.email)
    )
    manager = SyftboxManager.from_config(config)

    [built_client] = built
    for cached_client in _drive_cached_clients(manager._connection_router.transport):
        assert cached_client is built_client


def test_every_drive_component_shares_one_cached_client():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    transport = do_manager._connection_router.transport
    cached_client = drive_cached_client_of(transport)

    for component_cached_client in _drive_cached_clients(transport):
        assert component_cached_client is cached_client


def test_drive_cached_client_of_transport_not_on_drive_raises():
    transport = SyftboxManager._drive_transport(_mock_cached_client("do@example.com"))
    transport = transport.model_copy(update={"owner_store": Mock(spec=OwnerStore)})

    with pytest.raises(TypeError, match="not built on Google Drive"):
        drive_cached_client_of(transport)


def test_manager_without_transport_config_fails_at_init():
    with pytest.raises(ValueError, match="names no transport"):
        SyftboxManager.from_config(_config_with_both_roles())


@pytest.mark.parametrize(
    "p2p_config",
    [ConnectionConfig(), _drive_config("other@example.com")],
    ids=["not_drive", "other_drive_login"],
)
def test_transport_other_than_one_drive_connection_is_not_built_yet(p2p_config):
    config = _config_with_both_roles()
    config.transport_config = TransportConfig.one_connection(
        _drive_config(config.email)
    ).model_copy(update={"p2p": p2p_config})

    with pytest.raises(NotImplementedError, match="one Google Drive connection"):
        SyftboxManager.from_config(config)
