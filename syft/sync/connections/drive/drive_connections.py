"""The transport jobs on Google Drive.

Each class wraps a ``GDriveConnection`` and calls the method that does the job
today, so what is read from and written to Drive does not change. Where a
contract builds a method from simpler ones, the Drive class overrides it to
keep the single Drive call it makes today. The Drive code moves out of
``GDriveConnection`` into these classes, over a shared ``GDriveClient``, in a
later step.
"""

from __future__ import annotations

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Self

from syft.sync.connections.drive.gdrive_retry import raise_unavailable
from syft.sync.transport.exceptions import TransportUnavailableError
from syft.sync.transport.protocols import (
    Collections,
    OwnerStore,
    P2PConnection,
    Peering,
    Transport,
    WatcherStore,
)
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus

if TYPE_CHECKING:
    from syft.sync.connections.base_connection import FileCollection
    from syft.sync.connections.drive.gdrive_transport import GDriveConnection
    from syft.sync.version.peer_info import PeerInfo


# Threads that download the messages of one receive at the same time.
DOWNLOAD_WORKERS = 10


def _download_or_skip(connection: GDriveConnection, listed: dict) -> bytes | None:
    """The bytes of one listed inbox file, or None with a warning when this
    file cannot be downloaded, for example because it was deleted.

    Raises:
        TransportUnavailableError: when Drive cannot be reached.
    """
    name = listed["file_name"]
    try:
        with raise_unavailable(f"Drive is unreachable; inbox message {name} waits"):
            return connection.download_file(listed["file_id"])
    except TransportUnavailableError:
        raise
    except Exception as e:
        warnings.warn(f"Skipped inbox message {name}: download failed: {e!r}")
        return None


def _require_inbox_kind(kind: MessageKind) -> None:
    """Only the inbox is read through ``receive`` so far."""
    if kind is not MessageKind.PROPOSED_CHANGES:
        raise NotImplementedError(
            f"Messages of kind {kind!r} are still read through the outbox "
            "methods; they move to receive with send-and-forget."
        )


class DriveTransportComponent:
    """A transport component over one Google Drive connection."""

    def __init__(self, connection: GDriveConnection):
        self._conn = connection

    def clone_for_thread(self) -> Self:
        """A copy safe to use from another thread: the Drive client is not."""
        return type(self)(self._conn.clone_for_thread())

    def reset_caches(self) -> None:
        self._conn.reset_caches()


class DriveP2PConnection(DriveTransportComponent, P2PConnection):
    """Messages through the shared inbox and outbox folders.

    A proposed change goes to the data owner's inbox and file events go to the
    data scientist's outbox. ``msg_id`` is the file name. A written file is
    ``DELIVERED``: it waits in the folder until the peer reads it.

    The inbox is read through ``receive`` and a processed message is archived.
    The outbox is still read through the Drive-only methods below, from a
    timestamp cursor, and kept: the data scientist rebuilds a lost cache by
    reading it again. It moves to ``receive`` with send-and-forget, once the
    data scientist keeps what it received in its own storage.
    """

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

    def receive(
        self, peer: str, kind: MessageKind, max_messages: int | None = None
    ) -> list[RawMessage]:
        """One listing for the whole inbox, then the files download in parallel.

        A file that cannot be downloaded is left out with a warning, so it does
        not hold back the others.

        Raises:
            TransportUnavailableError: when Drive cannot be reached. Every
                message stays in the inbox.
        """
        _require_inbox_kind(kind)
        listed = self._conn.owner_get_inbox_file_metadatas(peer)[:max_messages]
        contents = self._download_all(listed)
        return [
            RawMessage(
                sender=peer,
                msg_id=m["file_name"],
                kind=kind,
                data=data,
                handle=m["file_id"],
            )
            for m, data in zip(listed, contents)
            if data is not None
        ]

    def _download_all(self, listed: list[dict]) -> list[bytes | None]:
        """The bytes of each listed file, in order; None for a file that could
        not be downloaded. Several files download over a thread pool with one
        connection copy per thread, because the Drive client is not
        thread-safe."""
        if len(listed) <= 1:
            return [_download_or_skip(self._conn, m) for m in listed]
        per_thread = threading.local()

        def download(m: dict) -> bytes | None:
            if not hasattr(per_thread, "conn"):
                per_thread.conn = self._conn.clone_for_thread()
            return _download_or_skip(per_thread.conn, m)

        workers = min(DOWNLOAD_WORKERS, len(listed))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(download, listed))

    def mark_processed(self, message: RawMessage) -> None:
        # The archive folder is named after ``message.sender``, which receive
        # set from the inbox folder, never from the message body.
        _require_inbox_kind(message.kind)
        self._conn.owner_archive_inbox_file(message.handle, message.sender)

    def mark_all_processed(self, messages: list[RawMessage]) -> None:
        for message in messages:
            _require_inbox_kind(message.kind)
        self._conn.owner_archive_inbox_files([(m.handle, m.sender) for m in messages])

    def outbox_messages(self, peer: str, since_timestamp: float | None) -> list[bytes]:
        """The messages in ``peer``'s outbox to me, newer than ``since_timestamp``."""
        return self._conn.watcher_download_raw_events_from_outbox(peer, since_timestamp)

    def outbox_message_metadatas(
        self, peer: str, since_timestamp: float | None
    ) -> list[dict]:
        """Like ``outbox_messages``, without downloading: each entry's
        ``file_id`` is the handle to ``fetch_message`` it with."""
        return self._conn.watcher_get_outbox_file_metadatas(peer, since_timestamp)

    def fetch_message(self, handle: str) -> bytes:
        return self._conn.download_file(handle)

    def own_outbox_message_metadatas(self, peer: str) -> list[dict]:
        """The messages in my outbox to ``peer``, without downloading. An
        outbox whose folder is not found holds none."""
        folder_id = self._conn._get_own_datasite_outbox_id(peer)
        if folder_id is None:
            return []
        return self._conn.get_file_metadatas_from_folder(
            folder_id, since_timestamp=None
        )

    def delete_messages(self, handles: list[str]) -> None:
        self._conn.delete_multiple_files_by_ids(handles)


class DriveOwnerStore(DriveTransportComponent, OwnerStore):
    """The data owner's own folders in /SyftBox: event log, checkpoints,
    rolling state."""

    def append_event(self, name: str, data: bytes) -> str:
        return self._conn.owner_write_raw_bytes_to_syftbox(name, data)

    def list_event_handles(self, since_timestamp: float | None = None) -> list[str]:
        return self._conn.owner_get_all_accepted_event_file_ids(since_timestamp)

    def fetch(self, handle: str) -> bytes:
        return self._conn.owner_download_raw_bytes_by_id(handle)

    def fetch_events_since(self, since_timestamp: float | None) -> list[bytes]:
        if since_timestamp is None:
            return self.fetch_all_events()
        return self._conn.download_raw_events_since_timestamp(since_timestamp)

    def fetch_all_events(self) -> list[bytes]:
        return self._conn.owner_download_all_raw_events_from_syftbox()

    def put_checkpoint(self, name: str, data: bytes) -> str:
        return self._conn.upload_raw_checkpoint(name, data)

    def latest_checkpoint(self) -> bytes | None:
        return self._conn.download_raw_latest_checkpoint()

    def put_incremental(self, name: str, data: bytes) -> str:
        return self._conn.upload_raw_incremental_checkpoint(name, data)

    def all_incrementals(self) -> list[bytes]:
        return self._conn.download_all_raw_incremental_checkpoints()

    def incremental_count(self) -> int:
        return self._conn.get_incremental_checkpoint_count()

    def next_incremental_sequence(self) -> int:
        return self._conn.get_next_incremental_sequence_number()

    def delete_incrementals(self) -> None:
        self._conn.delete_all_incremental_checkpoints()

    def put_rolling_state(self, name: str, data: bytes) -> str:
        return self._conn.upload_raw_rolling_state(name, data)

    def get_rolling_state(self) -> bytes | None:
        return self._conn.download_raw_rolling_state()

    def delete_rolling_state(self) -> None:
        self._conn.delete_rolling_state()

    def delete_all(self) -> int:
        """Everything in /SyftBox, and the syft files Drive moved to the root
        of My Drive when they lost their folder."""
        file_ids = set(self._conn.gather_all_file_and_folder_ids())
        file_ids |= set(self._conn.find_orphaned_syft_files())
        self._conn.delete_multiple_files_by_ids(list(file_ids))
        return len(file_ids)


class DriveWatcherStore(DriveTransportComponent, WatcherStore):
    """The data scientist's storage on Drive. Not built yet: today the data
    scientist keeps what it received in a local cache."""

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


class DriveCollections(DriveTransportComponent, Collections):
    """Collection folders in /SyftBox, shared by Drive permission."""

    def create(self, prefix: str, tag: str, content_hash: str, owner_email: str) -> str:
        return self._conn.owner_create_collection_folder(
            prefix, tag, content_hash, owner_email
        )

    def mark_shared_with_any(self, prefix: str, tag: str, content_hash: str) -> None:
        self._conn.owner_tag_collection_as_any(prefix, tag, content_hash)

    def grant_read(
        self, prefix: str, tag: str, content_hash: str, users: list[str]
    ) -> None:
        self._conn.owner_share_collection(prefix, tag, content_hash, users)

    def upload(
        self, prefix: str, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        self._conn.owner_upload_collection_files(prefix, tag, content_hash, files)

    def list_mine_with_permissions(self, prefix: str) -> list[FileCollection]:
        return self._conn.owner_list_all_collections_with_permissions(prefix)

    def list_mine(self, prefix: str) -> list[str]:
        return self._conn.owner_list_collections(prefix)

    def delete(self, prefix: str, tag: str) -> None:
        self._conn.owner_delete_collection(prefix, tag)

    def delete_by_handle(self, handle: str) -> None:
        self._conn.delete_file_by_id(handle)

    def list_shared_with_me(self, prefix: str) -> list[dict]:
        return self._conn.watcher_list_collections(prefix)

    def list_files(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> list[dict]:
        return self._conn.watcher_get_collection_file_metadatas(
            prefix, tag, content_hash, owner_email
        )

    def fetch_file(self, handle: str) -> bytes:
        return self._conn.watcher_download_collection_file(handle)

    def download(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        return self._conn.watcher_download_collection(
            prefix, tag, content_hash, owner_email
        )


class DrivePeering(DriveTransportComponent, Peering):
    """Identity from the Google account; channels, the peer list, key bundles
    and version files in /SyftBox."""

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


def drive_connection(transport: Transport) -> GDriveConnection:
    """The Drive connection under ``transport``, for tests and tools that need
    Drive itself. The sync code reaches Drive only through the components."""
    return transport.owner_store._conn
