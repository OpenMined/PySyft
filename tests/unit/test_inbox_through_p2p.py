"""Proposals go only through the manager's ``Transport.p2p``: with a fake p2p in
its place, a job reaches the data owner without one Drive inbox call. File
events still go through Drive, because the data scientist reads the outbox with
Drive-only calls until it moves to ``receive``."""

from syft.sync.connections.drive.drive_connections import DriveP2PConnection
from syft.sync.connections.drive.gdrive_transport import GDriveConnection
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.transport.protocols import P2PConnection
from syft.sync.transport.types import MessageKind, RawMessage, SendStatus
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access

DRIVE_INBOX_METHODS = [
    "watcher_send_raw_bytes_to_inbox",
    "owner_get_inbox_file_metadatas",
    "owner_archive_inbox_file",
    "owner_archive_inbox_files",
]


class InMemoryInbox(P2PConnection):
    """Proposals kept in ``messages`` by (sender, recipient), a dict every
    manager of a test shares. Every other message goes through ``drive``."""

    def __init__(
        self,
        email: str,
        drive: DriveP2PConnection,
        messages: dict[tuple[str, str], list[RawMessage]],
    ):
        self._email = email
        self._drive = drive
        self._messages = messages

    def send(
        self, peer: str, msg_id: str, data: bytes, kind: MessageKind
    ) -> SendStatus:
        if kind is not MessageKind.PROPOSED_CHANGES:
            return self._drive.send(peer, msg_id, data, kind)
        message = RawMessage(sender=self._email, msg_id=msg_id, kind=kind, data=data)
        self._messages.setdefault((self._email, peer), []).append(message)
        return SendStatus.DELIVERED

    def delivered(self) -> list[str]:
        return []

    def receive(
        self, peer: str, kind: MessageKind, max_messages: int | None = None
    ) -> list[RawMessage]:
        return list(self._messages.get((peer, self._email), []))[:max_messages]

    def mark_processed(self, message: RawMessage) -> None:
        self._messages[(message.sender, self._email)].remove(message)

    # The outbox, still read and compacted with Drive-only calls.
    def outbox_messages(self, peer: str, since_timestamp: float | None) -> list:
        return self._drive.outbox_messages(peer, since_timestamp)

    def own_outbox_message_metadatas(self, peer: str) -> list[dict]:
        return self._drive.own_outbox_message_metadatas(peer)

    def delete_messages(self, handles: list[str]) -> None:
        self._drive.delete_messages(handles)

    def outbox_message_metadatas(
        self, peer: str, since_timestamp: float | None
    ) -> list[dict]:
        return self._drive.outbox_message_metadatas(peer, since_timestamp)

    def clone_for_thread(self) -> DriveP2PConnection:
        # Only used to fetch outbox messages in parallel.
        return self._drive.clone_for_thread()


def _record_drive_inbox_calls(monkeypatch) -> list[str]:
    calls = []
    for name in DRIVE_INBOX_METHODS:
        original = getattr(GDriveConnection, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(GDriveConnection, name, spy)
    return calls


def test_job_reaches_do_through_transport_p2p_only(monkeypatch):
    drive_inbox_calls = _record_drive_inbox_calls(monkeypatch)
    messages = {}
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        make_p2p=lambda email, drive: InMemoryInbox(email, drive, messages)
    )
    grant_job_inbox_access(do_manager, ds_manager.email)

    job_path = path_for_job(do_manager.email, ds_manager.email)
    ds_manager._send_file_change(job_path, "Hello, world!")
    channel = (ds_manager.email, do_manager.email)
    assert len(messages[channel]) == 1

    do_manager.sync()
    ds_manager.sync()

    assert messages[channel] == []
    watcher_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert len(watcher_cache.get_cached_events()) == 1
    assert drive_inbox_calls == []
