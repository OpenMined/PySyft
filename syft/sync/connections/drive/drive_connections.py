"""The transport components on Google Drive.

Each component has its own module: ``drive_p2p``, ``drive_owner_store``,
``drive_collections`` and ``drive_peering``. They share one ``DriveLayout``,
which holds the Drive client. ``SyftboxManager._drive_transport`` builds them.
"""

from syft.sync.connections.drive.drive_owner_store import DriveOwnerStore
from syft.sync.connections.drive.gdrive_layout import DriveLayout
from syft.sync.transport.protocols import Transport, WatcherStore


class DriveWatcherStore(WatcherStore):
    """The DS's storage on Drive. Not built yet: today the DS keeps what it
    received in a local cache."""

    def put_checkpoint(self, peer: str, name: str, data: bytes) -> str:
        raise NotImplementedError("Watcher storage on Drive is not built yet")

    def latest_checkpoint(self, peer: str) -> bytes | None:
        raise NotImplementedError("Watcher storage on Drive is not built yet")

    def put_rolling_state(self, peer: str, name: str, data: bytes) -> str:
        raise NotImplementedError("Watcher storage on Drive is not built yet")

    def get_rolling_state(self, peer: str) -> bytes | None:
        raise NotImplementedError("Watcher storage on Drive is not built yet")

    def reset(self, peer: str) -> None:
        raise NotImplementedError("Watcher storage on Drive is not built yet")


def drive_layout_of(transport: Transport) -> DriveLayout:
    """The Drive layout under ``transport``, for tests and tools that need
    Drive itself. The sync code reaches Drive only through the components.
    Raises if the transport is not built on Drive."""
    store = transport.owner_store
    if not isinstance(store, DriveOwnerStore):
        raise TypeError("This transport is not built on Google Drive")
    return store._layout
