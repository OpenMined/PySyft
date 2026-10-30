"""Messages between peers on Google Drive: the inbox and outbox folders."""

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

from syft.sync.connections.drive.gdrive_client import DRIVE_BATCH_SIZE, GDriveClient
from syft.sync.connections.drive.drive_cached_client import (
    DriveCachedClient,
    valid_events_from_file_metadatas,
)
from syft.sync.connections.drive.gdrive_retry import (
    batch_execute_with_retries,
    execute_with_retries,
    raise_unavailable,
)
from syft.sync.events.file_change_event import FileChangeEventsMessageFileName
from syft.sync.messages.proposed_filechange import FileNameParseError, MessageFileName
from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.protocols import P2PConnection
from syft.sync.connections.types import MessageKind, RawMessage, SendStatus

# Threads that download the messages of one receive at the same time.
DOWNLOAD_WORKERS = 10


def _download_or_skip(client: GDriveClient, listed: dict) -> bytes | None:
    """The bytes of one listed message file, or None with a warning when this
    file cannot be downloaded, for example because it was deleted.

    Raises:
        ConnectionUnavailableError: when Drive cannot be reached.
    """
    name = listed["file_name"]
    try:
        with raise_unavailable(f"Drive is unreachable; message {name} waits"):
            return client.download_file(listed["file_id"])
    except ConnectionUnavailableError:
        raise
    except Exception as e:
        warnings.warn(f"Skipped message {name}: download failed: {e!r}")
        return None


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

    Both are read through ``receive``. A processed inbox message is archived.
    The outbox keeps every message, so the DS rebuilds a lost cache by reading
    it again, and ``receive`` lists only the messages after ``after``. That
    ends with send-and-forget, once the DS keeps what it received in its own
    storage. The DO still compacts its own outbox through the Drive-only
    methods below.
    """

    def __init__(self, cached_client: DriveCachedClient):
        self._cached_client = cached_client

    def clone_for_thread(self) -> "DriveP2PConnection":
        """A connection over a new Drive client, for use from another thread."""
        return DriveP2PConnection(self._cached_client.clone_for_thread())

    def reset_caches(self) -> None:
        self._cached_client.reset_caches()

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
        self,
        peer: str,
        kind: MessageKind,
        after: str | None = None,
        max_messages: int | None = None,
    ) -> list[RawMessage]:
        """One listing of the folder, then the files download in parallel.

        Proposed changes come from my inbox, which archives processed messages,
        so ``after`` is not needed there. File events come from ``peer``'s
        outbox, which keeps them, so only those after ``after`` are listed.

        A file that cannot be downloaded is left out with a warning, so it does
        not hold back the others.

        Raises:
            ConnectionUnavailableError: when Drive cannot be reached. Every
                message stays in its folder.
        """
        if kind is MessageKind.PROPOSED_CHANGES:
            listed = self._list_inbox(peer)
        else:
            listed = self._list_outbox(peer, after)
        listed = listed[:max_messages]
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
            return [_download_or_skip(self._cached_client.client, m) for m in listed]
        per_thread = threading.local()

        def download(m: dict) -> bytes | None:
            if not hasattr(per_thread, "client"):
                per_thread.client = self._cached_client.client.clone_for_thread()
            return _download_or_skip(per_thread.client, m)

        workers = min(DOWNLOAD_WORKERS, len(listed))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(download, listed))

    def mark_processed(self, message: RawMessage) -> None:
        # The outbox keeps file events; the DS passes ``after`` instead.
        if message.kind is MessageKind.FILE_EVENTS:
            return
        # The archive folder is named after ``message.sender``, which receive
        # set from the inbox folder, never from the message body.
        execute_with_retries(self._archive_request(message.handle, message.sender))

    def mark_all_processed(self, messages: list[RawMessage]) -> None:
        """Archive every inbox message, up to ``DRIVE_BATCH_SIZE`` moves per
        request. File events stay in the outbox.

        A move that fails is warned about and its file stays in the inbox.
        """
        messages = [m for m in messages if m.kind is MessageKind.PROPOSED_CHANGES]

        def callback(request_id, response, exception):
            if exception is not None:
                warnings.warn(
                    f"Could not archive inbox file {request_id}: {exception!r}"
                )

        drive_service = self._cached_client.client.drive_service
        for i in range(0, len(messages), DRIVE_BATCH_SIZE):
            batch = drive_service.new_batch_http_request(callback=callback)
            for m in messages[i : i + DRIVE_BATCH_SIZE]:
                batch.add(
                    self._archive_request(m.handle, m.sender), request_id=m.handle
                )
            batch_execute_with_retries(batch)

    def fetch_message(self, handle: str) -> bytes:
        return self._cached_client.client.download_file(handle)

    def own_outbox_message_metadatas(self, peer: str) -> list[dict]:
        """The messages in my outbox to ``peer``, without downloading. An
        outbox whose folder is not found holds none."""
        folder_id = self._cached_client.get_own_datasite_outbox_id(peer)
        if folder_id is None:
            return []
        return self._cached_client.get_file_metadatas_from_folder(
            folder_id, since_timestamp=None
        )

    def delete_messages(self, handles: list[str]) -> None:
        self._cached_client.client.delete_multiple_files_by_ids(handles)

    # -- Drive -----------------------------------------------------------------

    def _list_inbox(self, peer: str) -> list[dict]:
        """The messages in the inbox from ``peer``, oldest first, each with
        ``file_id`` and ``file_name``, without downloading them."""
        inbox_folder_id = self._cached_client.get_own_datasite_inbox_id(peer)
        if inbox_folder_id is None:
            raise ValueError(f"Inbox folder for {peer} not found")
        file_metadatas = self._cached_client.get_file_metadatas_from_folder(
            inbox_folder_id
        )
        name_to_id = {f["name"]: f["id"] for f in file_metadatas}
        return [
            {"file_id": name_to_id[name.as_string()], "file_name": name.as_string()}
            for name in sorted(
                _valid_message_names(file_metadatas),
                key=lambda x: x.submitted_timestamp,
            )
        ]

    def _list_outbox(self, peer: str, after: str | None) -> list[dict]:
        """The messages in ``peer``'s outbox to me sent after the message
        ``after``, oldest first, each with ``file_id`` and ``file_name``,
        without downloading them. An outbox whose folder is not found holds
        none."""
        folder_id = self._cached_client.get_peer_datasite_outbox_id(peer)
        if folder_id is None:
            return []
        since = None
        if after is not None:
            since = FileChangeEventsMessageFileName.from_string(after).timestamp
        file_metadatas = self._cached_client.get_file_metadatas_from_folder(
            folder_id, since_timestamp=since
        )
        name_to_id = {f["name"]: f["id"] for f in file_metadatas}
        names = sorted(
            valid_events_from_file_metadatas(file_metadatas), key=lambda x: x.timestamp
        )
        return [
            {"file_id": name_to_id[name.as_string()], "file_name": name.as_string()}
            for name in names
            if (since is None or name.timestamp > since)
            and name.as_string() in name_to_id
        ]

    def _archive_request(self, file_id: str, sender: str):
        """The request that moves an inbox file to its archive folder. The inbox
        folder is the file's only parent and both folder ids are cached, so the
        move is one Drive call. ``sender`` names the archive folder, so the
        caller reads it from the transport, never from the message body."""
        return self._cached_client.client.drive_service.files().update(
            fileId=file_id,
            addParents=self._cached_client.get_archive_folder_id(sender),
            removeParents=self._cached_client.get_own_datasite_inbox_id(sender),
            fields="id",
            supportsAllDrives=True,
        )

    def watcher_send_raw_bytes_to_inbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        inbox_id = self._cached_client.get_peer_datasite_inbox_id(recipient)
        if inbox_id is None:
            raise Exception(f"Inbox folder for {recipient}'s datasite not found")

        payload, _ = self._cached_client.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [inbox_id]}

        result = self._cached_client.client.create_syft_file(
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
                self._cached_client.client.drive_service.files().update(
                    fileId=file_id,
                    addParents=inbox_id,
                    fields="id, parents",
                )
            )

    def owner_write_raw_bytes_to_outbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        outbox_folder_id = self._cached_client.get_own_datasite_outbox_id(recipient)
        if outbox_folder_id is None:
            raise ValueError(f"Outbox folder for {recipient} not found")

        file_payload, _ = self._cached_client.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [outbox_folder_id]}

        result = self._cached_client.client.create_syft_file(
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
                self._cached_client.client.drive_service.files().update(
                    fileId=file_id,
                    addParents=outbox_folder_id,
                    fields="id, parents",
                )
            )
