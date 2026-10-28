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
from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.transport.protocols import (
    P2PConnection,
    Peering,
    Transport,
    WatcherStore,
)
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus

if TYPE_CHECKING:
    from syft.sync.connections.drive.gdrive_transport import GDriveConnection
    from syft.sync.version.peer_info import PeerInfo


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


class DrivePeering(Peering):
    """Identity from the Google account; channels, the peer list, key bundles
    and version files in /SyftBox."""

    def __init__(self, connection: GDriveConnection):
        self._conn = connection

    def whoami(self) -> str:
        return self._conn.get_authenticated_email()

    def open_channels(self, peer: str) -> None:
        self._conn.add_peer(peer_email=peer)

    def incoming_requests(self) -> list[str]:
        return self._conn.get_peer_requests()

    def read_peer_registry(self, force_download: bool = False) -> dict:
        return self._conn._get_peers_json(force_download=force_download)

    def update_peer_registry(
        self, peer: str, state: str, public_encryption_bundle: dict | None = None
    ) -> None:
        self._conn._update_peer_state(peer, state, public_encryption_bundle)

    def publish_key_bundle(self, peer: str, bundle_json: str) -> None:
        self._conn.write_encryption_bundle(peer, bundle_json)

    def grant_key_bundle_read(self, peer: str) -> None:
        self._conn.share_encryption_bundles_folder(peer)

    def read_own_key_bundle(self, peer: str) -> str | None:
        return self._conn.read_own_encryption_bundle(peer)

    def read_peer_key_bundle(self, peer: str) -> str | None:
        return self._conn.read_peer_encryption_bundle(peer)

    def publish_peer_info(self, peer_info: PeerInfo) -> None:
        self._conn.write_peer_info_file(peer_info)

    def read_own_peer_info(self) -> PeerInfo | None:
        return self._conn.read_own_peer_info_file()

    def read_peer_info(self, peer: str) -> PeerInfo | None:
        return self._conn.read_peer_info_file(peer)

    def grant_peer_info_read(self, peer: str, peer_info: PeerInfo) -> None:
        self._conn.share_peer_info_file_with_peer(peer, peer_info)


def drive_transport(connection: GDriveConnection) -> Transport:
    """The transport jobs over one Drive connection."""
    return Transport(
        p2p=DriveP2PConnection(connection),
        owner_store=DriveOwnerStore(connection.layout),
        watcher_store=DriveWatcherStore(connection),
        collections=DriveCollections(connection.layout),
        peering=DrivePeering(connection),
        platform=GdriveFilesPlatform(),
    )
