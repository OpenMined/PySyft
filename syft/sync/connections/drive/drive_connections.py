"""The router's connections on Google Drive.

Each connection has its own module: ``drive_p2p``, ``drive_owner_store``,
``drive_watcher_store``, ``drive_collections`` and ``drive_peering``. They share one ``DriveCachedClient``,
which holds the Drive client. ``SyftboxManager._drive_connections`` builds them.
"""

from syft.sync.connections.drive.drive_owner_store import DriveOwnerStoreConnection
from syft.sync.connections.drive.drive_cached_client import DriveCachedClient
from syft.sync.connections.protocols import Connections


def drive_cached_client_of(connections: Connections) -> DriveCachedClient:
    """The ``DriveCachedClient`` under ``connections``, for tests and tools that
    need Drive itself. The sync code reaches Drive only through the components.
    Raises if the connections are not built on Drive."""
    store = connections.owner_store_con
    if not isinstance(store, DriveOwnerStoreConnection):
        raise TypeError("These connections are not built on Google Drive")
    return store._cached_client
