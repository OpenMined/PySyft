"""The transport jobs on Google Drive, and the router that calls them.

The Drive classes wrap today's GDriveConnection methods, so these tests pin
what each job promises, on the mock Drive service the rest of the suite uses.
"""

import inspect
import uuid
from unittest.mock import Mock

import pytest

import syft.sync.connections.connection_router as connection_router_module
from syft.sync.sync import datasite_owner_syncer
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.connections.drive.drive_connections import DriveP2PConnection
from syft.sync.connections.drive.gdrive_transport import (
    CollectionFolder,
    GDriveConnection,
)
from syft.sync.connections.drive.mock_drive_service import pair_with_mock_service
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.events.file_change_event import FileChangeEventsMessage
from syft.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft.sync.peers.peer_store import PeerStore
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
    ds_conn.as_transport().peering.open_channels(DO)
    return do_conn, ds_conn


# -- the transport ----------------------------------------------------------------


def test_drive_transport_does_every_job():
    do_conn, _ = pair_with_mock_service(DO, DS)
    transport = do_conn.as_transport()
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
    status = ds_conn.as_transport().p2p.send(
        DO, _proposal_name(1.0), b"job", MessageKind.PROPOSED_CHANGES
    )
    assert status is SendStatus.DELIVERED
    data, _ = do_conn.as_transport().p2p.next_inbox_message(DS)
    assert data == b"job"


def test_an_archived_inbox_message_is_not_read_again(pair):
    do_conn, ds_conn = pair
    ds_conn.as_transport().p2p.send(
        DO, _proposal_name(1.0), b"job", MessageKind.PROPOSED_CHANGES
    )
    p2p = do_conn.as_transport().p2p

    _, handle = p2p.next_inbox_message(DS)
    p2p.archive_inbox_message(DS, handle)
    assert p2p.next_inbox_message(DS) is None


def test_file_events_sent_through_p2p_reach_the_scientist_outbox(pair):
    do_conn, ds_conn = pair
    name = _events_name(1.0)
    status = do_conn.as_transport().p2p.send(
        DS, name, b"results", MessageKind.FILE_EVENTS
    )
    assert status is SendStatus.DELIVERED
    p2p = ds_conn.as_transport().p2p
    listed = p2p.outbox_message_metadatas(DO, since_timestamp=None)
    assert [m["file_name"] for m in listed] == [name]
    assert p2p.fetch_message(listed[0]["file_id"]) == b"results"
    assert p2p.outbox_messages(DO, since_timestamp=None) == [b"results"]


def test_a_p2p_copy_for_another_thread_reads_the_same_outbox(pair):
    do_conn, ds_conn = pair
    do_conn.as_transport().p2p.send(
        DS, _events_name(1.0), b"results", MessageKind.FILE_EVENTS
    )
    p2p = ds_conn.as_transport().p2p
    [listed] = p2p.outbox_message_metadatas(DO, since_timestamp=None)

    copy = p2p.clone_for_thread()
    assert copy._conn is not ds_conn
    assert copy.fetch_message(listed["file_id"]) == b"results"


def test_an_unknown_message_kind_is_refused(pair):
    do_conn, _ = pair
    with pytest.raises(ValueError):
        do_conn.as_transport().p2p.send(DS, _events_name(1.0), b"x", "unknown-kind")


def test_drive_never_has_sends_waiting_for_confirmation(pair):
    do_conn, _ = pair
    assert do_conn.as_transport().p2p.delivered() == []


def test_receiving_through_p2p_is_not_built_yet(pair):
    do_conn, _ = pair
    p2p = do_conn.as_transport().p2p
    message = RawMessage(
        sender=DS, msg_id="m", kind=MessageKind.PROPOSED_CHANGES, data=b""
    )
    with pytest.raises(NotImplementedError):
        p2p.receive()
    with pytest.raises(NotImplementedError):
        p2p.ack(message)


# -- OwnerStore -------------------------------------------------------------------


def test_owner_event_log_is_read_oldest_first(pair):
    do_conn, _ = pair
    store = do_conn.as_transport().owner_store
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
    store = do_conn.as_transport().owner_store
    store.append_event(_events_name(1.0), b"event")
    store.append_event("syfteventsmessagev3_not-a-valid-name.tar.gz", b"junk")

    assert store.count_events_since(None) == len(store.list_event_handles()) == 1


def test_owner_store_keeps_the_newest_full_checkpoint(pair):
    do_conn, _ = pair
    store = do_conn.as_transport().owner_store
    store.put_checkpoint("checkpoint_1.0.tar.gz", b"one")
    store.put_checkpoint("checkpoint_2.0.tar.gz", b"two")
    assert store.latest_checkpoint() == b"two"


def test_owner_store_counts_and_deletes_incremental_checkpoints(pair):
    do_conn, _ = pair
    store = do_conn.as_transport().owner_store
    store.put_incremental("incremental_checkpoint_1_1.0.tar.gz", b"inc")
    assert store.incremental_count() == 1
    assert store.next_incremental_sequence() == 2
    assert store.all_incrementals() == [b"inc"]
    store.delete_incrementals()
    assert store.incremental_count() == 0


def test_owner_store_overwrites_and_deletes_the_rolling_state(pair):
    do_conn, _ = pair
    store = do_conn.as_transport().owner_store
    store.put_rolling_state("rolling_state_1.0.tar.gz", b"state 1")
    store.put_rolling_state("rolling_state_2.0.tar.gz", b"state 2")
    assert store.get_rolling_state() == b"state 2"
    store.delete_rolling_state()
    assert store.get_rolling_state() is None


def test_owner_store_files_carry_the_syft_tag(pair):
    do_conn, _ = pair
    store = do_conn.as_transport().owner_store
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
        ds_conn.as_transport().watcher_store.latest_checkpoint(DO)


# -- Collections ------------------------------------------------------------------


def test_a_published_collection_is_listed_and_downloaded_by_its_reader(pair):
    do_conn, ds_conn = pair
    owner, reader = do_conn.as_transport(), ds_conn.as_transport()
    files = {"mock.csv": b"a,b\n1,2\n"}
    content_hash = CollectionFolder.compute_hash(files)

    owner.collections.create(DATASETS, "census", content_hash, DO)
    owner.collections.upload(DATASETS, "census", content_hash, files)
    owner.collections.grant_read(DATASETS, "census", content_hash, [DS])

    assert owner.collections.list_mine(DATASETS) == ["census"]
    mine = owner.collections.list_mine_with_permissions(DATASETS)
    assert [c.tag for c in mine] == ["census"]

    shared = reader.collections.list_shared_with_me(DATASETS)
    assert [(c["owner_email"], c["tag"]) for c in shared] == [(DO, "census")]
    assert reader.collections.download(DATASETS, "census", content_hash, DO) == files
    [listed] = reader.collections.list_files(DATASETS, "census", content_hash, DO)
    assert reader.collections.fetch_file(listed["file_id"]) == files["mock.csv"]

    owner.collections.delete(DATASETS, "census")
    assert owner.collections.list_mine(DATASETS) == []


# -- Peering ----------------------------------------------------------------------


def test_peering_finds_the_peer_that_opened_channels(pair):
    do_conn, _ = pair
    peering = do_conn.as_transport().peering
    assert peering.whoami() == DO
    assert DS in peering.incoming_requests()


def test_peering_keeps_the_peer_registry(pair):
    do_conn, _ = pair
    peering = do_conn.as_transport().peering
    peering.update_peer_registry(DS, "accepted")
    assert peering.read_peer_registry(force_download=True)[DS]["state"] == "accepted"


def test_peer_registry_update_while_drive_is_down_keeps_other_peers(pair, monkeypatch):
    do_conn, _ = pair
    peering = do_conn.as_transport().peering
    peering.update_peer_registry(DS, "accepted")
    do_conn._peers_json_cache = None  # a new process has no cache

    monkeypatch.setattr(GDriveConnection, "download_file", _drive_down)
    with pytest.raises(TransportUnavailableError):
        peering.update_peer_registry("other@example.com", "accepted")
    monkeypatch.undo()

    assert peering.read_peer_registry(force_download=True)[DS]["state"] == "accepted"


def test_reading_own_key_bundle_creates_no_folder(pair):
    owner = pair[0].as_transport()
    assert owner.peering.read_own_key_bundle(DS) is None
    assert pair[0]._find_encryption_bundles_folder_id() is None


def test_peering_publishes_a_key_bundle_both_sides_can_read(pair):
    do_conn, ds_conn = pair
    owner, scientist = do_conn.as_transport(), ds_conn.as_transport()
    bundle = '{"bundle": "do-public-key"}'

    owner.peering.publish_key_bundle(DS, bundle)
    owner.peering.grant_key_bundle_read(DS)

    assert scientist.peering.read_peer_key_bundle(DO) == bundle
    assert owner.peering.read_own_key_bundle(DS) == bundle
    assert owner.peering.read_own_key_bundle("someone-else@example.com") is None


def test_peering_publishes_peer_info_the_peer_can_read(pair):
    do_conn, ds_conn = pair
    owner, scientist = do_conn.as_transport(), ds_conn.as_transport()
    info = PeerInfo(
        version_info=VersionInfo.current(),
        encryption_policy=EncryptionPolicy.REQUIRED,
    )

    owner.peering.publish_peer_info(info)
    owner.peering.grant_peer_info_read(DS, info)

    assert owner.peering.read_own_peer_info() == info
    assert scientist.peering.read_peer_info(DO) == info


# -- ConnectionRouter -------------------------------------------------------------


def _router_with_mock_p2p():
    do_conn, _ = pair_with_mock_service(DO, DS)
    router = ConnectionRouter(connections=[do_conn], peer_store=_store_knowing(DO, DS))
    drive = do_conn.as_transport()
    p2p = Mock(spec=DriveP2PConnection)
    router._transport = drive.model_copy(update={"p2p": p2p})
    router._transport_source = do_conn
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
    p2p.next_inbox_message.return_value = None
    assert router.owner_get_next_proposed_filechange_message(DS) is None
    p2p.next_inbox_message.assert_called_once_with(DS)


def test_the_router_archives_a_handled_message_through_p2p():
    router, p2p = _router_with_mock_p2p()
    message = ProposedFileChangesMessage(sender_email=DS, proposed_file_changes=[])
    router.owner_remove_proposed_filechange_from_inbox(message, DS)
    p2p.archive_proposed_changes.assert_called_once_with(message, DS)


def test_the_router_reads_the_outbox_through_p2p():
    router, p2p = _router_with_mock_p2p()
    p2p.outbox_messages.return_value = []
    p2p.outbox_message_metadatas.return_value = []
    assert router.watcher_get_events_messages(DO, 5.0) == []
    assert router.watcher_get_outbox_file_metadatas(DO, 5.0) == []
    p2p.outbox_messages.assert_called_once_with(DO, 5.0)
    p2p.outbox_message_metadatas.assert_called_once_with(DO, 5.0)


def test_the_router_reuses_its_transport_until_its_connection_changes():
    do_conn, other_conn = pair_with_mock_service(DO, DS)
    router = ConnectionRouter(connections=[do_conn], peer_store=PeerStore(email=DO))
    first = router.transport
    assert router.transport is first

    router.connections[0] = other_conn
    assert router.transport is not first


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
