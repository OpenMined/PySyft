"""The transport jobs on Google Drive.

Each class wraps a ``GDriveConnection`` and calls the method that does the job
today, so what is read from and written to Drive does not change. Where a
contract builds a method from simpler ones, the Drive class overrides it to
keep the single Drive call it makes today. The Drive code moves out of
``GDriveConnection`` into these classes, over a shared ``GDriveClient``, in a
later step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from syft.sync.connections.drive.drive_collections import DriveCollections
from syft.sync.connections.drive.drive_owner_store import DriveOwnerStore
from syft.sync.connections.drive.drive_p2p import DriveP2PConnection
from syft.sync.connections.drive.drive_peering import DrivePeering
from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.transport.protocols import (
    Transport,
    WatcherStore,
)

if TYPE_CHECKING:
    from syft.sync.connections.drive.gdrive_transport import GDriveConnection


class DriveWatcherStore(WatcherStore):
    """The data scientist's storage on Drive. Not built yet: today the data
    scientist keeps what it received in a local cache."""

    def __init__(self, connection: GDriveConnection):
        self._conn = connection

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


def drive_transport(connection: GDriveConnection) -> Transport:
    """The transport jobs over one Drive connection."""
    return Transport(
        p2p=DriveP2PConnection(connection.layout),
        owner_store=DriveOwnerStore(connection.layout),
        watcher_store=DriveWatcherStore(connection),
        collections=DriveCollections(connection.layout),
        peering=DrivePeering(connection.layout),
        platform=GdriveFilesPlatform(),
    )
