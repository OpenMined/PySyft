"""Messages between peers on Google Drive: the inbox and outbox folders."""

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

from syft.sync.connections.drive.gdrive_client import DRIVE_BATCH_SIZE, GDriveClient
from syft.sync.connections.drive.gdrive_layout import (
    DriveLayout,
    valid_events_from_file_metadatas,
)
from syft.sync.connections.drive.gdrive_retry import (
    batch_execute_with_retries,
    execute_with_retries,
    raise_unavailable,
)
from syft.sync.messages.proposed_filechange import FileNameParseError, MessageFileName
from syft.sync.transport.exceptions import TransportUnavailableError
from syft.sync.transport.protocols import P2PConnection
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus

# Threads that download the messages of one receive at the same time.
DOWNLOAD_WORKERS = 10


def _download_or_skip(client: GDriveClient, listed: dict) -> bytes | None:
    """The bytes of one listed inbox file, or None with a warning when this
    file cannot be downloaded, for example because it was deleted.

    Raises:
        TransportUnavailableError: when Drive cannot be reached.
    """
    name = listed["file_name"]
    try:
        with raise_unavailable(f"Drive is unreachable; inbox message {name} waits"):
            return client.download_file(listed["file_id"])
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


def _valid_message_names(file_metadatas: list[dict]) -> list[MessageFileName]:
    res = []
    for file_metadata in file_metadatas:
        try:
            res.append(MessageFileName.from_string(file_metadata["name"]))
        except FileNameParseError:
            continue
    return res


class DriveP2PConnection(P2PConnection):
    """Messages through the shared inbox and outbox folders.

    A proposed change goes to the DO's inbox and file events go to the DS's
    outbox. ``msg_id`` is the file name. A written file is ``DELIVERED``: it
    waits in the folder until the peer reads it.

    The inbox is read through ``receive`` and a processed message is archived.
    The outbox is still read through the Drive-only methods below, from a
    timestamp cursor, and kept: the DS rebuilds a lost cache by reading it
    again. It moves to ``receive`` with send-and-forget, once the DS keeps what
    it received in its own storage.
    """

    def __init__(self, layout: DriveLayout):
        self._layout = layout

    def clone_for_thread(self) -> "DriveP2PConnection":
        """A connection over a new Drive client, for use from another thread."""
        return DriveP2PConnection(self._layout.clone_for_thread())

    def reset_caches(self) -> None:
        self._layout.reset_caches()

    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        if kind is MessageKind.PROPOSED_CHANGES:
            self.watcher_send_raw_bytes_to_inbox(peer, msg_id, data)
        elif kind is MessageKind.FILE_EVENTS:
            self.owner_write_raw_bytes_to_outbox(peer, msg_id, data)
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
        listed = self._list_inbox(peer)[:max_messages]
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
        client copy per thread, because the Drive client is not thread-safe."""
        if len(listed) <= 1:
            return [_download_or_skip(self._layout.client, m) for m in listed]
        per_thread = threading.local()

        def download(m: dict) -> bytes | None:
            if not hasattr(per_thread, "client"):
                per_thread.client = self._layout.client.clone_for_thread()
            return _download_or_skip(per_thread.client, m)

        workers = min(DOWNLOAD_WORKERS, len(listed))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(download, listed))

    def mark_processed(self, message: RawMessage) -> None:
        # The archive folder is named after ``message.sender``, which receive
        # set from the inbox folder, never from the message body.
        _require_inbox_kind(message.kind)
        execute_with_retries(self._archive_request(message.handle, message.sender))

    def mark_all_processed(self, messages: list[RawMessage]) -> None:
        """Archive every message, up to ``DRIVE_BATCH_SIZE`` moves per request.

        A move that fails is warned about and its file stays in the inbox.
        """
        for message in messages:
            _require_inbox_kind(message.kind)

        def callback(request_id, response, exception):
            if exception is not None:
                warnings.warn(
                    f"Could not archive inbox file {request_id}: {exception!r}"
                )

        drive_service = self._layout.client.drive_service
        for i in range(0, len(messages), DRIVE_BATCH_SIZE):
            batch = drive_service.new_batch_http_request(callback=callback)
            for m in messages[i : i + DRIVE_BATCH_SIZE]:
                batch.add(
                    self._archive_request(m.handle, m.sender), request_id=m.handle
                )
            batch_execute_with_retries(batch)

    def outbox_messages(self, peer: str, since_timestamp: float | None) -> list[bytes]:
        """The messages in ``peer``'s outbox to me, newer than ``since_timestamp``."""
        return [
            self._layout.client.download_file(m["file_id"])
            for m in self.outbox_message_metadatas(peer, since_timestamp)
        ]

    def outbox_message_metadatas(
        self, peer: str, since_timestamp: float | None
    ) -> list[dict]:
        """Like ``outbox_messages``, without downloading: each entry's
        ``file_id`` is the handle to ``fetch_message`` it with."""
        folder_id = self._layout.get_peer_datasite_outbox_id(peer)
        if folder_id is None:
            return []

        file_metadatas = self._layout.get_file_metadatas_from_folder(
            folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = valid_events_from_file_metadatas(file_metadatas)
        name_to_id = {f["name"]: f["id"] for f in file_metadatas}

        result = []
        for fname_obj in sorted(valid_fname_objs, key=lambda x: x.timestamp):
            if since_timestamp is None or fname_obj.timestamp > since_timestamp:
                file_name = fname_obj.as_string()
                if file_name in name_to_id:
                    result.append(
                        {
                            "file_id": name_to_id[file_name],
                            "file_name": file_name,
                            "timestamp": fname_obj.timestamp,
                        }
                    )
        return result

    def fetch_message(self, handle: str) -> bytes:
        return self._layout.client.download_file(handle)

    def own_outbox_message_metadatas(self, peer: str) -> list[dict]:
        """The messages in my outbox to ``peer``, without downloading. An
        outbox whose folder is not found holds none."""
        folder_id = self._layout.get_own_datasite_outbox_id(peer)
        if folder_id is None:
            return []
        return self._layout.get_file_metadatas_from_folder(
            folder_id, since_timestamp=None
        )

    def delete_messages(self, handles: list[str]) -> None:
        self._layout.client.delete_multiple_files_by_ids(handles)

    # -- Drive -----------------------------------------------------------------

    def _list_inbox(self, peer: str) -> list[dict]:
        """The messages in the inbox from ``peer``, oldest first, each with
        ``file_id`` and ``file_name``, without downloading them."""
        inbox_folder_id = self._layout.get_own_datasite_inbox_id(peer)
        if inbox_folder_id is None:
            raise ValueError(f"Inbox folder for {peer} not found")
        file_metadatas = self._layout.get_file_metadatas_from_folder(inbox_folder_id)
        name_to_id = {f["name"]: f["id"] for f in file_metadatas}
        return [
            {"file_id": name_to_id[name.as_string()], "file_name": name.as_string()}
            for name in sorted(
                _valid_message_names(file_metadatas),
                key=lambda x: x.submitted_timestamp,
            )
        ]

    def _archive_request(self, file_id: str, sender: str):
        """The request that moves an inbox file to its archive folder. The inbox
        folder is the file's only parent and both folder ids are cached, so the
        move is one Drive call. ``sender`` names the archive folder, so the
        caller reads it from the transport, never from the message body."""
        return self._layout.client.drive_service.files().update(
            fileId=file_id,
            addParents=self._layout.get_archive_folder_id(sender),
            removeParents=self._layout.get_own_datasite_inbox_id(sender),
            fields="id",
            supportsAllDrives=True,
        )

    def watcher_send_raw_bytes_to_inbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        inbox_id = self._layout.get_peer_datasite_inbox_id(recipient)
        if inbox_id is None:
            raise Exception(f"Inbox folder for {recipient}'s datasite not found")

        payload, _ = self._layout.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [inbox_id]}

        result = self._layout.client.create_syft_file(
            file_metadata, payload, fields="id, parents"
        )

        file_id = result.get("id")
        actual_parents = result.get("parents", [])
        if inbox_id not in actual_parents:
            print(
                f"WARNING: Message file {file_id} was not placed in inbox folder "
                f"{inbox_id}. Actual parents: {actual_parents}. "
                f"Moving file to correct folder..."
            )
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=file_id,
                    addParents=inbox_id,
                    fields="id, parents",
                )
            )

    def owner_write_raw_bytes_to_outbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        outbox_folder_id = self._layout.get_own_datasite_outbox_id(recipient)
        if outbox_folder_id is None:
            raise ValueError(f"Outbox folder for {recipient} not found")

        file_payload, _ = self._layout.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [outbox_folder_id]}

        result = self._layout.client.create_syft_file(
            file_metadata, file_payload, fields="id, parents"
        )

        file_id = result.get("id")
        actual_parents = result.get("parents", [])
        if outbox_folder_id not in actual_parents:
            print(
                f"WARNING: Event message {file_id} was not placed in outbox folder "
                f"{outbox_folder_id}. Actual parents: {actual_parents}. "
                f"Moving file to correct folder..."
            )
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=file_id,
                    addParents=outbox_folder_id,
                    fields="id, parents",
                )
            )
