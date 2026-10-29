"""The transport jobs on Google Drive, and the router that calls them.

The Drive classes wrap today's GDriveConnection methods, so these tests pin
what each job promises, on the mock Drive service the rest of the suite uses.
"""

import inspect
import threading
import time
import uuid
from collections import Counter
from unittest.mock import Mock

import pytest

import syft.sync.connections.connection_router as connection_router_module
from syft.sync.sync import datasite_owner_syncer
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.connections.drive.drive_connections import (
    DriveCollections,
    DriveOwnerStore,
    DriveP2PConnection,
    DrivePeering,
    DriveWatcherStore,
)
from syft.sync.connections.drive.gdrive_transport import (
    CollectionFolder,
    GDriveConnection,
)
from syft.sync.connections.drive import mock_drive_service
from syft.sync.connections.drive.mock_drive_service import pair_with_mock_service
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.events.file_change_event import FileChangeEventsMessage
from syft.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.transport import (
    Collections,
    MessageKind,
    OwnerStore,
    P2PConnection,
    Peering,
    RawMessage,
    SendStatus,
    Transport,
    TransportUnavailableError,
    WatcherStore,
)
from syft.sync.version.peer_info import PeerInfo
from syft.sync.version.version_info import VersionInfo
from tests.unit.test_sync_manager import _store_knowing

DO = "do@example.com"
DS = "ds@example.com"
DATASETS = "syft_datasetcollection"


def _drive_down(self, file_id: str) -> bytes:
    raise ConnectionResetError("Drive is down")


def _proposal_name(ts: float) -> str:
    return f"msgv2_{ts}_{uuid.uuid4().hex}.tar.gz"


def _events_name(ts: float) -> str:
    return f"syfteventsmessagev3_{ts}_{uuid.uuid4()}.tar.gz"


@pytest.fixture
def pair():
    """DO and DS connections over one mock Drive, with the DS's channels open."""
    do_conn, ds_conn = pair_with_mock_service(DO, DS)
    DrivePeering(ds_conn).open_channels(DO)
    return do_conn, ds_conn


# -- the transport ----------------------------------------------------------------


def test_drive_transport_does_every_job():
    do_conn, _ = pair_with_mock_service(DO, DS)
    transport = SyftboxManager._drive_transport(do_conn)
    assert isinstance(transport, Transport)
    assert isinstance(transport.p2p, P2PConnection)
    assert isinstance(transport.owner_store, OwnerStore)
    assert isinstance(transport.watcher_store, WatcherStore)
    assert isinstance(transport.collections, Collections)
    assert isinstance(transport.peering, Peering)
    assert isinstance(transport.platform, GdriveFilesPlatform)


# -- P2PConnection ----------------------------------------------------------------


def test_a_proposed_change_sent_through_p2p_reaches_the_owner_inbox(pair):
    do_conn, ds_conn = pair
    status = DriveP2PConnection(ds_conn).send(
        DO, _proposal_name(1.0), b"job", MessageKind.PROPOSED_CHANGES
    )
    assert status is SendStatus.DELIVERED
    [message] = DriveP2PConnection(do_conn).receive(DS, MessageKind.PROPOSED_CHANGES)
    assert (message.sender, message.kind, message.data) == (
        DS,
        MessageKind.PROPOSED_CHANGES,
        b"job",
    )


def test_inbox_messages_are_received_oldest_first_up_to_max(pair):
    do_conn, ds_conn = pair
    names = {submitted: _proposal_name(submitted) for submitted in (2.0, 1.0, 3.0)}
    for submitted, name in names.items():
        DriveP2PConnection(ds_conn).send(
            DO, name, f"job {submitted}".encode(), MessageKind.PROPOSED_CHANGES
        )
    p2p = DriveP2PConnection(do_conn)

    received = p2p.receive(DS, MessageKind.PROPOSED_CHANGES, max_messages=2)
    assert [(m.msg_id, m.data) for m in received] == [
        (names[1.0], b"job 1.0"),
        (names[2.0], b"job 2.0"),
    ]


def _count_drive_calls(monkeypatch) -> Counter:
    """Count mock Drive calls by kind: list, get, update, batch and get_media.
    A batch request also counts each update inside it."""
    calls = Counter()
    lock = threading.Lock()

    def count(kind: str) -> None:
        with lock:
            calls[kind] += 1

    for kind, request_class in (
        ("list", mock_drive_service.MockListRequest),
        ("get", mock_drive_service.MockGetRequest),
        ("update", mock_drive_service.MockUpdateRequest),
        ("batch", mock_drive_service.MockBatchHttpRequest),
    ):
        original = request_class.execute

        def counted(self, *args, _kind=kind, _original=original, **kwargs):
            count(_kind)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(request_class, "execute", counted)
    original_get_media = mock_drive_service.MockFilesResource.get_media

    def counted_get_media(self, *args, **kwargs):
        count("get_media")
        return original_get_media(self, *args, **kwargs)

    monkeypatch.setattr(
        mock_drive_service.MockFilesResource, "get_media", counted_get_media
    )
    return calls


def _send_proposals(ds_conn, count: int) -> list[str]:
    names = [_proposal_name(float(i)) for i in range(count)]
    for name in names:
        DriveP2PConnection(ds_conn).send(
            DO, name, name.encode(), MessageKind.PROPOSED_CHANGES
        )
    return names


def test_receiving_inbox_lists_once_and_downloads_each_message(pair, monkeypatch):
    do_conn, ds_conn = pair
    p2p = DriveP2PConnection(do_conn)
    p2p.receive(DS, MessageKind.PROPOSED_CHANGES)  # finds the inbox folder once
    names = _send_proposals(ds_conn, 3)

    calls = _count_drive_calls(monkeypatch)
    received = p2p.receive(DS, MessageKind.PROPOSED_CHANGES)

    assert [(m.msg_id, m.data) for m in received] == [(n, n.encode()) for n in names]
    assert calls == Counter(list=1, get_media=3)


def test_inbox_messages_download_at_the_same_time(pair, monkeypatch):
    do_conn, ds_conn = pair
    _send_proposals(ds_conn, 3)
    running, most_running = [0], [0]
    lock = threading.Lock()
    original_download = GDriveConnection.download_file

    def slow_download(self, file_id: str) -> bytes:
        with lock:
            running[0] += 1
            most_running[0] = max(most_running[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return original_download(self, file_id)

    monkeypatch.setattr(GDriveConnection, "download_file", slow_download)
    DriveP2PConnection(do_conn).receive(DS, MessageKind.PROPOSED_CHANGES)
    assert most_running[0] > 1


def test_marking_inbox_message_processed_is_one_drive_call(pair, monkeypatch):
    do_conn, ds_conn = pair
    _send_proposals(ds_conn, 2)
    p2p = DriveP2PConnection(do_conn)
    first, second = p2p.receive(DS, MessageKind.PROPOSED_CHANGES)
    p2p.mark_processed(first)  # finds or creates the archive folder once

    calls = _count_drive_calls(monkeypatch)
    p2p.mark_processed(second)

    assert calls == Counter(update=1)
    assert p2p.receive(DS, MessageKind.PROPOSED_CHANGES) == []


def test_marking_inbox_messages_processed_is_one_batch_request(pair, monkeypatch):
    do_conn, ds_conn = pair
    _send_proposals(ds_conn, 4)
    p2p = DriveP2PConnection(do_conn)
    first, *rest = p2p.receive(DS, MessageKind.PROPOSED_CHANGES)
    p2p.mark_processed(first)  # finds or creates the archive folder once

    calls = _count_drive_calls(monkeypatch)
    p2p.mark_all_processed(rest)

    assert calls == Counter(batch=1, update=3)  # one request holding three moves
    assert p2p.receive(DS, MessageKind.PROPOSED_CHANGES) == []


def test_move_that_fails_in_batch_warns_and_rest_are_archived(pair):
    do_conn, ds_conn = pair
    _send_proposals(ds_conn, 2)
    p2p = DriveP2PConnection(do_conn)
    first, second = p2p.receive(DS, MessageKind.PROPOSED_CHANGES)
    gone = first.model_copy(update={"handle": "no-such-file"})

    with pytest.warns(UserWarning, match="Could not archive inbox file no-such-file"):
        p2p.mark_all_processed([gone, second])

    assert [m.msg_id for m in p2p.receive(DS, MessageKind.PROPOSED_CHANGES)] == [
        first.msg_id
    ]


def test_inbox_message_that_fails_to_download_is_skipped_with_warning(
    pair, monkeypatch
):
    do_conn, ds_conn = pair
    names = _send_proposals(ds_conn, 3)
    [broken_id] = [
        m["file_id"]
        for m in do_conn.owner_get_inbox_file_metadatas(DS)
        if m["file_name"] == names[1]
    ]
    original_download = GDriveConnection.download_file

    def download_all_but_broken(self, file_id: str) -> bytes:
        if file_id == broken_id:
            raise ValueError("file is gone")
        return original_download(self, file_id)

    monkeypatch.setattr(GDriveConnection, "download_file", download_all_but_broken)
    with pytest.warns(UserWarning, match=f"Skipped inbox message {names[1]}"):
        received = DriveP2PConnection(do_conn).receive(DS, MessageKind.PROPOSED_CHANGES)

    assert [m.msg_id for m in received] == [names[0], names[2]]


def test_receiving_inbox_while_drive_is_down_raises(pair, monkeypatch):
    do_conn, ds_conn = pair
    _send_proposals(ds_conn, 2)
    monkeypatch.setattr(GDriveConnection, "download_file", _drive_down)
    with pytest.raises(TransportUnavailableError):
        DriveP2PConnection(do_conn).receive(DS, MessageKind.PROPOSED_CHANGES)


def test_processed_inbox_message_is_not_received_again(pair):
    do_conn, ds_conn = pair
    DriveP2PConnection(ds_conn).send(
        DO, _proposal_name(1.0), b"job", MessageKind.PROPOSED_CHANGES
    )
    p2p = DriveP2PConnection(do_conn)

    [message] = p2p.receive(DS, MessageKind.PROPOSED_CHANGES)
    p2p.mark_processed(message)
    assert p2p.receive(DS, MessageKind.PROPOSED_CHANGES) == []


def test_file_events_sent_through_p2p_reach_the_scientist_outbox(pair):
    do_conn, ds_conn = pair
    name = _events_name(1.0)
    status = DriveP2PConnection(do_conn).send(
        DS, name, b"results", MessageKind.FILE_EVENTS
    )
    assert status is SendStatus.DELIVERED
    p2p = DriveP2PConnection(ds_conn)
    listed = p2p.outbox_message_metadatas(DO, since_timestamp=None)
    assert [m["file_name"] for m in listed] == [name]
    assert p2p.fetch_message(listed[0]["file_id"]) == b"results"
    assert p2p.outbox_messages(DO, since_timestamp=None) == [b"results"]


def test_a_p2p_copy_for_another_thread_reads_the_same_outbox(pair):
    do_conn, ds_conn = pair
    DriveP2PConnection(do_conn).send(
        DS, _events_name(1.0), b"results", MessageKind.FILE_EVENTS
    )
    p2p = DriveP2PConnection(ds_conn)
    [listed] = p2p.outbox_message_metadatas(DO, since_timestamp=None)

    copy = p2p.clone_for_thread()
    assert copy._conn is not ds_conn
    assert copy.fetch_message(listed["file_id"]) == b"results"


def test_an_unknown_message_kind_is_refused(pair):
    do_conn, _ = pair
    with pytest.raises(ValueError):
        DriveP2PConnection(do_conn).send(DS, _events_name(1.0), b"x", "unknown-kind")


def test_drive_never_has_sends_waiting_for_confirmation(pair):
    do_conn, _ = pair
    assert DriveP2PConnection(do_conn).delivered() == []


def test_receiving_file_events_through_p2p_is_not_built_yet(pair):
    _, ds_conn = pair
    p2p = DriveP2PConnection(ds_conn)
    message = RawMessage(sender=DO, msg_id="m", kind=MessageKind.FILE_EVENTS, data=b"")
    with pytest.raises(NotImplementedError):
        p2p.receive(DO, MessageKind.FILE_EVENTS)
    with pytest.raises(NotImplementedError):
        p2p.mark_processed(message)


# -- OwnerStore -------------------------------------------------------------------


def test_owner_event_log_is_read_oldest_first(pair):
    do_conn, _ = pair
    store = DriveOwnerStore(do_conn)
    # Written out of order: Drive lists by name, newest first.
    second = store.append_event(_events_name(2.0), b"second")
    first = store.append_event(_events_name(1.0), b"first")

    assert store.list_event_handles() == [first, second]
    assert store.fetch_all_events() == [b"first", b"second"]
    assert store.fetch_events_since(None) == [b"first", b"second"]
    assert store.fetch_events_since(1.5) == [b"second"]
    assert store.fetch(first) == b"first"
    assert store.count_events_since(None) == 2
    assert store.count_events_since(1.5) == 1


def test_owner_event_count_matches_the_listed_handles(pair):
    do_conn, _ = pair
    store = DriveOwnerStore(do_conn)
    store.append_event(_events_name(1.0), b"event")
    store.append_event("syfteventsmessagev3_not-a-valid-name.tar.gz", b"junk")

    assert store.count_events_since(None) == len(store.list_event_handles()) == 1


def test_owner_store_keeps_the_newest_full_checkpoint(pair):
    do_conn, _ = pair
    store = DriveOwnerStore(do_conn)
    store.put_checkpoint("checkpoint_1.0.tar.gz", b"one")
    store.put_checkpoint("checkpoint_2.0.tar.gz", b"two")
    assert store.latest_checkpoint() == b"two"


def test_owner_store_counts_and_deletes_incremental_checkpoints(pair):
    do_conn, _ = pair
    store = DriveOwnerStore(do_conn)
    store.put_incremental("incremental_checkpoint_1_1.0.tar.gz", b"inc")
    assert store.incremental_count() == 1
    assert store.next_incremental_sequence() == 2
    assert store.all_incrementals() == [b"inc"]
    store.delete_incrementals()
    assert store.incremental_count() == 0


def test_owner_store_overwrites_and_deletes_the_rolling_state(pair):
    do_conn, _ = pair
    store = DriveOwnerStore(do_conn)
    store.put_rolling_state("rolling_state_1.0.tar.gz", b"state 1")
    store.put_rolling_state("rolling_state_2.0.tar.gz", b"state 2")
    assert store.get_rolling_state() == b"state 2"
    store.delete_rolling_state()
    assert store.get_rolling_state() is None


def test_owner_store_files_carry_the_syft_tag(pair):
    do_conn, _ = pair
    store = DriveOwnerStore(do_conn)
    store.put_checkpoint("checkpoint_1.0.tar.gz", b"full")
    store.put_incremental("incremental_checkpoint_1_1.0.tar.gz", b"inc")
    store.put_rolling_state("rolling_state_1.0.tar.gz", b"state")

    do_files = [
        f
        for f in do_conn.drive_service._backing_store.files.values()
        if f.owners[0]["emailAddress"] == DO
    ]
    untagged = [f.name for f in do_files if f.properties != {"syft": "true"}]
    assert do_files
    assert untagged == []


# -- WatcherStore -----------------------------------------------------------------


def test_watcher_storage_on_drive_is_not_built_yet(pair):
    _, ds_conn = pair
    with pytest.raises(NotImplementedError):
        DriveWatcherStore(ds_conn).latest_checkpoint(DO)


# -- Collections ------------------------------------------------------------------


def test_a_published_collection_is_listed_and_downloaded_by_its_reader(pair):
    do_conn, ds_conn = pair
    owner, reader = DriveCollections(do_conn), DriveCollections(ds_conn)
    files = {"mock.csv": b"a,b\n1,2\n"}
    content_hash = CollectionFolder.compute_hash(files)

    owner.create(DATASETS, "census", content_hash, DO)
    owner.upload(DATASETS, "census", content_hash, files)
    owner.grant_read(DATASETS, "census", content_hash, [DS])

    assert owner.list_mine(DATASETS) == ["census"]
    mine = owner.list_mine_with_permissions(DATASETS)
    assert [c.tag for c in mine] == ["census"]

    shared = reader.list_shared_with_me(DATASETS)
    assert [(c["owner_email"], c["tag"]) for c in shared] == [(DO, "census")]
    assert reader.download(DATASETS, "census", content_hash, DO) == files
    [listed] = reader.list_files(DATASETS, "census", content_hash, DO)
    assert reader.fetch_file(listed["file_id"]) == files["mock.csv"]

    owner.delete(DATASETS, "census")
    assert owner.list_mine(DATASETS) == []


# -- Peering ----------------------------------------------------------------------


def test_peering_finds_the_peer_that_opened_channels(pair):
    do_conn, _ = pair
    peering = DrivePeering(do_conn)
    assert peering.whoami() == DO
    assert DS in peering.incoming_requests()


def test_peering_keeps_the_peer_registry(pair):
    do_conn, _ = pair
    peering = DrivePeering(do_conn)
    peering.update_peer_registry(DS, "accepted")
    assert peering.read_peer_registry(force_download=True)[DS]["state"] == "accepted"


def test_peer_registry_update_while_drive_is_down_keeps_other_peers(pair, monkeypatch):
    do_conn, _ = pair
    peering = DrivePeering(do_conn)
    peering.update_peer_registry(DS, "accepted")
    do_conn._peers_json_cache = None  # a new process has no cache

    monkeypatch.setattr(GDriveConnection, "download_file", _drive_down)
    with pytest.raises(TransportUnavailableError):
        peering.update_peer_registry("other@example.com", "accepted")
    monkeypatch.undo()

    assert peering.read_peer_registry(force_download=True)[DS]["state"] == "accepted"


def test_reading_own_key_bundle_creates_no_folder(pair):
    owner = DrivePeering(pair[0])
    assert owner.read_own_key_bundle(DS) is None
    assert pair[0]._find_encryption_bundles_folder_id() is None


def test_peering_publishes_a_key_bundle_both_sides_can_read(pair):
    do_conn, ds_conn = pair
    owner, scientist = DrivePeering(do_conn), DrivePeering(ds_conn)
    bundle = '{"bundle": "do-public-key"}'

    owner.publish_key_bundle(DS, bundle)
    owner.grant_key_bundle_read(DS)

    assert scientist.read_peer_key_bundle(DO) == bundle
    assert owner.read_own_key_bundle(DS) == bundle
    assert owner.read_own_key_bundle("someone-else@example.com") is None


def test_peering_publishes_peer_info_the_peer_can_read(pair):
    do_conn, ds_conn = pair
    owner, scientist = DrivePeering(do_conn), DrivePeering(ds_conn)
    info = PeerInfo(
        version_info=VersionInfo.current(),
        encryption_policy=EncryptionPolicy.REQUIRED,
    )

    owner.publish_peer_info(info)
    owner.grant_peer_info_read(DS, info)

    assert owner.read_own_peer_info() == info
    assert scientist.read_peer_info(DO) == info


# -- ConnectionRouter -------------------------------------------------------------


def _router_with_mock_p2p():
    do_conn, _ = pair_with_mock_service(DO, DS)
    p2p = Mock(spec=DriveP2PConnection)
    transport = SyftboxManager._drive_transport(do_conn).model_copy(update={"p2p": p2p})
    router = ConnectionRouter(transport=transport, peer_store=_store_knowing(DO, DS))
    return router, p2p


def test_the_router_sends_file_events_through_p2p():
    router, p2p = _router_with_mock_p2p()
    message = FileChangeEventsMessage(events=[])
    router.owner_write_event_messages_to_outbox(DS, message)

    peer, msg_id, _, kind = p2p.send.call_args.args
    assert (peer, msg_id, kind) == (
        DS,
        message.message_filepath.as_string(),
        MessageKind.FILE_EVENTS,
    )


def test_the_router_sends_proposed_changes_through_p2p():
    router, p2p = _router_with_mock_p2p()
    message = ProposedFileChangesMessage(sender_email=DO, proposed_file_changes=[])
    router.watcher_send_proposed_file_changes_message(DS, message)

    peer, msg_id, _, kind = p2p.send.call_args.args
    assert (peer, msg_id, kind) == (
        DS,
        message.message_filename.as_string(),
        MessageKind.PROPOSED_CHANGES,
    )


def test_the_router_reads_the_inbox_through_p2p():
    router, p2p = _router_with_mock_p2p()
    p2p.receive.return_value = []
    assert router.owner_receive_proposed_messages(DS) == []
    p2p.receive.assert_called_once_with(DS, MessageKind.PROPOSED_CHANGES)


def test_the_router_archives_unreadable_proposal_and_returns_the_rest():
    router, p2p = _router_with_mock_p2p()
    proposal = ProposedFileChangesMessage(sender_email=DS, proposed_file_changes=[])
    unreadable = RawMessage(
        sender=DS, msg_id="bad", kind=MessageKind.PROPOSED_CHANGES, data=b"garbage"
    )
    readable = RawMessage(
        sender=DS,
        msg_id="good",
        kind=MessageKind.PROPOSED_CHANGES,
        data=proposal.as_compressed_data(),
    )
    p2p.receive.return_value = [unreadable, readable]

    with pytest.warns(UserWarning, match="Archived message bad unread"):
        pairs = router.owner_receive_proposed_messages(DS)

    assert [(raw.msg_id, message.id) for raw, message in pairs] == [
        ("good", proposal.id)
    ]
    p2p.mark_processed.assert_called_once_with(unreadable)


def test_the_router_marks_handled_messages_processed_through_p2p():
    router, p2p = _router_with_mock_p2p()
    message = RawMessage(
        sender=DS, msg_id="m", kind=MessageKind.PROPOSED_CHANGES, data=b""
    )
    router.owner_mark_all_processed([message])
    p2p.mark_all_processed.assert_called_once_with([message])


def test_the_router_reads_the_outbox_through_p2p():
    router, p2p = _router_with_mock_p2p()
    p2p.outbox_messages.return_value = []
    p2p.outbox_message_metadatas.return_value = []
    assert router.watcher_get_events_messages(DO, 5.0) == []
    assert router.watcher_get_outbox_file_metadatas(DO, 5.0) == []
    p2p.outbox_messages.assert_called_once_with(DO, 5.0)
    p2p.outbox_message_metadatas.assert_called_once_with(DO, 5.0)


def test_the_router_holds_no_drive_specific_code():
    source = inspect.getsource(connection_router_module).lower()
    assert "gdrive" not in source
    assert "drive_service" not in source


# -- restoring the owner's state --------------------------------------------------


def test_restoring_from_the_event_log_keeps_the_newest_version(monkeypatch):
    """A fresh client that restores from the event log alone replays the
    messages oldest first, so a file changed twice ends at its last version."""
    # At 1 the event log is not written.
    monkeypatch.setattr(
        datasite_owner_syncer, "DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD", 2
    )
    _, do = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )
    files = do.datasite_owner_syncer.event_cache.file_connection
    files.write_file("public/report.txt", "version 1")
    do.sync()
    files.write_file("public/report.txt", "version 2")
    do.sync()

    router = do._connection_router
    router.delete_rolling_state()
    router.delete_all_incremental_checkpoints()

    fresh = do._copy()
    fresh.datasite_owner_syncer.pull_initial_state()
    restored = fresh.datasite_owner_syncer.event_cache.file_connection.read_file(
        "public/report.txt"
    )
    assert restored == "version 2"
