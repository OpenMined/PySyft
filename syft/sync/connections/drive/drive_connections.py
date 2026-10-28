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
from syft.sync.connections.drive.drive_peering import DrivePeering
from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.transport.protocols import (
    P2PConnection,
    Transport,
    WatcherStore,
)
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus

if TYPE_CHECKING:
    from syft.sync.connections.drive.gdrive_transport import GDriveConnection


class DriveP2PConnection(P2PConnection):
    """Messages through the shared inbox and outbox folders.

    A proposed change goes to the data owner's inbox and file events go to the
    data scientist's outbox. ``msg_id`` is the file name. A written file is
    ``DELIVERED``: it waits in the folder until the peer reads it.

    Only sending goes through this class so far. The router still reads and
    archives messages with the Drive methods directly, because doing it through
    ``receive`` and ``ack`` changes how they behave (one rule for both folders,
    delete on ack, no read cursor); that lands with send-and-forget.
    """

    def __init__(self, connection: GDriveConnection):
        self._conn = connection

    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        if kind is MessageKind.PROPOSED_CHANGES:
            self._conn.watcher_send_raw_bytes_to_inbox(peer, msg_id, data)
        elif kind is MessageKind.FILE_EVENTS:
            self._conn.owner_write_raw_bytes_to_outbox(peer, msg_id, data)
        else:
            raise ValueError(f"Drive has no folder for messages of kind {kind!r}")
        return SendStatus.DELIVERED

    def delivered(self) -> list[str]:
        # Every Drive send is DELIVERED when it returns; none waits for a peer.
        return []

    def receive(self, max_messages: int | None = None) -> list[RawMessage]:
        raise NotImplementedError(
            "Messages are still read through the Drive methods; receiving "
            "through P2PConnection lands with send-and-forget."
        )

    def ack(self, message: RawMessage) -> None:
        raise NotImplementedError(
            "Messages are still archived through the Drive methods; "
            "acknowledging through P2PConnection lands with send-and-forget."
        )


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
        p2p=DriveP2PConnection(connection),
        owner_store=DriveOwnerStore(connection.layout),
        watcher_store=DriveWatcherStore(connection),
        collections=DriveCollections(connection.layout),
        peering=DrivePeering(connection.layout),
        platform=GdriveFilesPlatform(),
    )
