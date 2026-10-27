"""The data owner sends each peer what it lacks, and deletes what it may no
longer have, from a per-peer record of what the peer holds."""

import threading
from pathlib import Path

import pytest

from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.peers.peer import PeerState
from syft.sync.sync import datasite_owner_syncer
from syft.sync.sync.constants import CACHE_DIR
from syft.sync.sync.caches.datasite_owner_cache import (
    ProposedEventFileOutdatedException,
)
from syft.sync.sync.delivery_tracker import (
    DELIVERY_STATE_VERSION,
    DELIVERY_VIEWS_FILENAME,
    DeliveryState,
    DeliveryTracker,
    LocalDeliveryStateStore,
    MemoryDeliveryStateStore,
)
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.transport.types import SendStatus
from syft_perms import SyftPermContext

PEER = "ds@example.com"


def _tracker_holding(view: dict[str, str]) -> DeliveryTracker:
    tracker = DeliveryTracker(MemoryDeliveryStateStore())
    tracker.record_send(PEER, "m0", SendStatus.DELIVERED, view)
    return tracker


def test_plan_sends_changed_files_deletes_unreadable():
    tracker = _tracker_holding({"same": "h1", "old": "h1", "revoked": "h1"})
    plan = tracker.plan({PEER: {"same": "h1", "old": "h2", "new": "h3"}})[PEER]
    assert plan.send == ["new", "old"]
    assert plan.delete == ["revoked"]


def test_delivered_send_updates_view():
    tracker = _tracker_holding({"a": "h1", "b": "h1"})
    tracker.record_send(PEER, "m1", SendStatus.DELIVERED, {"a": "h2", "b": None})
    assert tracker.plan({PEER: {"a": "h2"}})[PEER].send == []
    assert tracker.plan({PEER: {"a": "h2"}})[PEER].delete == []


def test_seed_runs_once_and_keeps_recorded_views():
    tracker = _tracker_holding({"b": "h9"})
    tracker.seed({PEER: {"a": "h1", "b": "h1"}})
    tracker.seed({PEER: {}})
    assert tracker.plan({PEER: {"a": "h1", "b": "h9"}})[PEER].send == []


def test_reset_resends_readable_deletes_gone():
    tracker = _tracker_holding({"a": "h1", "gone": "h1"})
    tracker.reset(PEER)
    plan = tracker.plan({PEER: {"a": "h1"}})[PEER]
    assert plan.send == ["a"]
    assert plan.delete == ["gone"]


def test_queued_send_updates_view_on_confirm():
    tracker = DeliveryTracker(MemoryDeliveryStateStore())
    tracker.record_send(PEER, "m1", SendStatus.QUEUED, {"a": "h1"})
    assert tracker.plan({PEER: {"a": "h1"}})[PEER].send == ["a"]
    tracker.confirm_deliveries(["m1"])
    assert tracker.plan({PEER: {"a": "h1"}})[PEER].send == []


def test_local_store_persists_state_refuses_later_version(tmp_path):
    path = tmp_path / DELIVERY_VIEWS_FILENAME
    DeliveryTracker(LocalDeliveryStateStore(path)).record_send(
        PEER, "m1", SendStatus.DELIVERED, {"a": "h1"}
    )
    assert (
        DeliveryTracker(LocalDeliveryStateStore(path))
        .plan({PEER: {"a": "h1"}})[PEER]
        .send
        == []
    )

    path.write_text(DeliveryState(version=DELIVERY_STATE_VERSION + 1).model_dump_json())
    with pytest.raises(ValueError, match="version"):
        LocalDeliveryStateStore(path).load()


def test_overlapping_writers_keep_both_views(tmp_path, monkeypatch):
    path = tmp_path / DELIVERY_VIEWS_FILENAME
    first_loaded = threading.Event()
    go = threading.Event()
    load = LocalDeliveryStateStore.load

    def load_then_wait_in_first_writer(self):
        state = load(self)
        if threading.current_thread().name == "first":
            first_loaded.set()
            go.wait(timeout=5)
        return state

    monkeypatch.setattr(LocalDeliveryStateStore, "load", load_then_wait_in_first_writer)

    def send(peer: str):
        tracker = DeliveryTracker(LocalDeliveryStateStore(path))
        tracker.record_send(peer, f"m-{peer}", SendStatus.DELIVERED, {"a": "h1"})

    first = threading.Thread(target=send, args=("one@x.com",), name="first")
    second = threading.Thread(target=send, args=("two@x.com",), name="second")
    first.start()
    first_loaded.wait(timeout=5)
    second.start()
    # Without the lock, the second writer saves now, and the first writer then
    # overwrites it with the state it read before.
    second.join(timeout=0.5)
    go.set()
    first.join()
    second.join()

    plans = DeliveryTracker(LocalDeliveryStateStore(path)).plan(
        {"one@x.com": {"a": "h1"}, "two@x.com": {"a": "h1"}}
    )
    assert plans["one@x.com"].send == plans["two@x.com"].send == []


# --- The owner syncer ---------------------------------------------------------


def _pair(**kwargs):
    return SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False, **kwargs
    )


def _write(do_manager, path: str, content: str):
    file = do_manager.syftbox_folder / do_manager.email / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)


def _received(ds_manager, do_manager, path: str):
    return ds_manager.syftbox_folder / do_manager.email / path


def _outbox_paths(ds_manager, do_manager) -> list[str]:
    messages = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    return [str(e.path_in_datasite) for m in messages for e in m.events]


def _sent_events(ds_manager, do_manager, path: str) -> list[tuple[bool, object]]:
    """(is_deleted, content) of each event for ``path`` in the DS's outbox."""
    messages = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    return [
        (e.is_deleted, e.content)
        for m in messages
        for e in m.events
        if str(e.path_in_datasite) == path
    ]


def _lose_views(do_manager) -> None:
    """Delete the DO's views file, keeping its file hashes."""
    (do_manager.syftbox_folder / CACHE_DIR / DELIVERY_VIEWS_FILENAME).unlink()


def _push_private_file(do_manager, ds_manager) -> None:
    """Send a file outside the plan, as a private dataset share does."""
    syncer = do_manager.datasite_owner_syncer
    _write(do_manager, "private/data.csv", "rows")
    message = syncer.event_cache.create_events_for_files(
        {Path("private/data.csv"): b"rows"}
    )
    syncer.queue_event_for_syftbox(
        recipients=[ds_manager.email], file_change_events_message=message
    )
    syncer.process_syftbox_events_queue()


def test_peer_approved_later_gets_existing_files():
    ds_manager, do_manager = _pair(add_peers=False)
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/data.txt", "existing")
    do_manager.sync(auto_checkpoint=False)

    ds_manager.add_peer(do_manager.email, sync=False)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    ds_manager.load_peers()
    do_manager.sync(auto_checkpoint=False)
    ds_manager.sync()

    assert (
        _received(ds_manager, do_manager, "shared/data.txt").read_text() == "existing"
    )


@pytest.mark.parametrize(
    "use_in_memory_cache, persist_owner_state",
    [(False, True), (False, False), (True, False)],
)
def test_revoked_peer_gets_content_then_delete(
    use_in_memory_cache, persist_owner_state
):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=use_in_memory_cache, persist_owner_state=persist_owner_state
    )
    syncer = do_manager.datasite_owner_syncer
    folder = syncer.perm_context.open("shared/")
    folder.grant_read_access(ds_manager.email)
    syncer.event_cache.file_connection.write_file("shared/data.txt", "secret")

    do_manager.sync(auto_checkpoint=False)
    do_manager.sync(auto_checkpoint=False)
    folder.revoke_read_access(ds_manager.email)
    do_manager.sync(auto_checkpoint=False)
    assert _sent_events(ds_manager, do_manager, "shared/data.txt") == [
        (False, "secret"),
        (True, None),
    ]


def test_rejected_then_approved_peer_gets_every_file_and_deletes():
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/data.txt", "one")
    _write(do_manager, "shared/gone.txt", "one")
    do_manager.sync(auto_checkpoint=False)

    # The DS asks again. The DO rejects it, deletes a file, and then approves a
    # later request.
    peer = do_manager.peer_manager.get_cached_peer(ds_manager.email)
    peer.state = PeerState.REQUESTED_BY_PEER
    do_manager.reject_peer_request(ds_manager.email)
    (do_manager.syftbox_folder / do_manager.email / "shared/gone.txt").unlink()
    peer.state = PeerState.REQUESTED_BY_PEER
    do_manager.approve_peer_request(ds_manager.email)

    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager).count("shared/data.txt") == 2
    assert _sent_events(ds_manager, do_manager, "shared/gone.txt")[-1] == (True, None)


def test_restarted_owner_sends_nothing_new():
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/data.txt", "one")
    do_manager.sync(auto_checkpoint=False)
    sent_before = _outbox_paths(ds_manager, do_manager)

    do_manager._copy().sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager) == sent_before


def test_lost_views_sends_only_offline_changes():
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/held.txt", "one")
    do_manager.sync(auto_checkpoint=False)

    _lose_views(do_manager)
    _write(do_manager, "shared/offline.txt", "new")

    do_manager.sync(auto_checkpoint=False)
    sent = _outbox_paths(ds_manager, do_manager)
    assert sent.count("shared/held.txt") == 1
    assert sent.count("shared/offline.txt") == 1


def test_lost_views_seeded_after_send_outside_plan():
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/held.txt", "one")
    do_manager.sync(auto_checkpoint=False)
    _lose_views(do_manager)

    _push_private_file(do_manager, ds_manager)
    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager).count("shared/held.txt") == 1


def test_lost_views_seeds_peer_skipped_in_first_sync():
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/held.txt", "one")
    do_manager.sync(auto_checkpoint=False)
    _lose_views(do_manager)

    # The version gate skips the DS in the first sync after the views are lost.
    do_manager.datasite_owner_syncer.sync([])
    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager).count("shared/held.txt") == 1


def test_send_to_peer_not_ready_dropped(monkeypatch):
    ds_manager, do_manager = _pair()

    def peer_not_ready(self, recipient, *args, **kwargs):
        raise SyftPeerNotReadyError(recipient, "keys not known yet", "sync later")

    with monkeypatch.context() as m:
        m.setattr(
            ConnectionRouter, "owner_write_event_messages_to_outbox", peer_not_ready
        )
        with pytest.warns(UserWarning, match="are not sent"):
            _push_private_file(do_manager, ds_manager)

    do_manager.datasite_owner_syncer.process_syftbox_events_queue()
    assert _sent_events(ds_manager, do_manager, "private/data.csv") == []


def test_peer_not_ready_does_not_block_other_peers():
    ds_manager, do_manager = _pair()
    syncer = do_manager.datasite_owner_syncer
    not_ready = "stranger@example.com"
    shared = syncer.perm_context.open("shared/")
    shared.grant_read_access(not_ready)
    shared.grant_read_access(ds_manager.email)
    _write(do_manager, "shared/data.txt", "one")

    syncer._record_local_changes()
    with pytest.warns(UserWarning, match=not_ready):
        syncer.deliver_file_changes([not_ready, ds_manager.email])
    assert _outbox_paths(ds_manager, do_manager) == ["shared/data.txt"]
    # The skipped peer's view did not change, so its next plan sends the file.
    readable = syncer._readable_files_by_peer([not_ready])
    assert syncer.event_cache.deliveries.plan(readable)[not_ready].send == [
        "shared/data.txt"
    ]


def test_failed_send_retried_then_logged_without_blocking_others(monkeypatch, caplog):
    ds_manager, do_manager = _pair()
    syncer = do_manager.datasite_owner_syncer
    broken = "broken@example.com"
    shared = syncer.perm_context.open("shared/")
    shared.grant_read_access(broken)
    shared.grant_read_access(ds_manager.email)
    _write(do_manager, "shared/data.txt", "one")
    syncer._record_local_changes()

    send = ConnectionRouter.owner_write_event_messages_to_outbox

    def fail_for_broken(self, recipient, message):
        if recipient == broken:
            raise ConnectionError("upload failed")
        return send(self, recipient, message)

    monkeypatch.setattr(
        ConnectionRouter, "owner_write_event_messages_to_outbox", fail_for_broken
    )
    syncer.deliver_file_changes([broken, ds_manager.email])
    assert _outbox_paths(ds_manager, do_manager) == ["shared/data.txt"]
    tries = [r for r in caplog.records if broken in r.getMessage()]
    assert [r.levelname for r in tries] == ["INFO", "INFO", "ERROR"]


def test_send_to_stranger_does_not_block_later_syncs():
    ds_manager, do_manager = _pair()
    syncer = do_manager.datasite_owner_syncer
    _write(do_manager, "private/data.csv", "rows")
    message = syncer.event_cache.create_events_for_files(
        {Path("private/data.csv"): b"rows"}
    )
    syncer.queue_event_for_syftbox(
        recipients=["stranger@example.com"], file_change_events_message=message
    )
    with pytest.warns(UserWarning, match="stranger@example.com"):
        syncer.process_syftbox_events_queue()

    syncer.perm_context.open("shared/").grant_read_access(ds_manager.email)
    _write(do_manager, "shared/data.txt", "one")
    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager) == ["shared/data.txt"]


def test_private_dataset_sent_to_peer_never_revoked():
    ds_manager, do_manager = _pair()
    _push_private_file(do_manager, ds_manager)

    do_manager.sync(auto_checkpoint=False)
    do_manager.sync(auto_checkpoint=False)
    assert _sent_events(ds_manager, do_manager, "private/data.csv") == [
        (False, b"rows")
    ]


def test_sync_reloads_permissions_once(monkeypatch):
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/data.txt", "one")
    reload = SyftPermContext._reload
    reloads = []

    def counting_reload(self):
        reloads.append(self)
        reload(self)

    monkeypatch.setattr(SyftPermContext, "_reload", counting_reload)
    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager) == ["shared/data.txt"]
    assert len(reloads) == 1


def test_large_catch_up_split_into_messages(monkeypatch):
    monkeypatch.setattr(datasite_owner_syncer, "MAX_DELIVERY_MESSAGE_CONTENT", 6)
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    for name in ("a", "b", "c"):
        _write(do_manager, f"shared/{name}.txt", "abc")

    do_manager.sync(auto_checkpoint=False)
    messages = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    assert sorted(len(m.events) for m in messages) == [1, 2]


def test_outdated_proposal_logged_without_blocking_owner_changes(caplog):
    ds_manager, do_manager = _pair()
    shared = do_manager.datasite_owner_syncer.perm_context.open("shared/")
    shared.grant_write_access(ds_manager.email)
    _write(do_manager, "shared/a.txt", "v1")
    do_manager.sync(auto_checkpoint=False)
    ds_manager.sync()

    # The DS edits a.txt from v1 while the DO edits it too, so the
    # proposal is outdated and its handling fails on every sync.
    ds_manager._send_file_change(f"{do_manager.email}/shared/a.txt", "ds edit")
    _write(do_manager, "shared/a.txt", "do edit")
    _write(do_manager, "shared/b.txt", "new")
    do_manager.sync(auto_checkpoint=False)
    assert ProposedEventFileOutdatedException.__name__ in caplog.text
    assert (
        do_manager.syftbox_folder / do_manager.email / "shared/a.txt"
    ).read_text() == "do edit"

    ds_manager.sync()
    assert _received(ds_manager, do_manager, "shared/b.txt").read_text() == "new"


def test_failed_send_not_retried_after_revoke(monkeypatch):
    monkeypatch.setattr(datasite_owner_syncer, "MAX_DELIVERY_MESSAGE_CONTENT", 3)
    ds_manager, do_manager = _pair()
    folder = do_manager.datasite_owner_syncer.perm_context.open("shared/")
    folder.grant_read_access(ds_manager.email)
    _write(do_manager, "shared/a.txt", "abc")
    _write(do_manager, "shared/b.txt", "abc")

    def fail_upload(*args, **kwargs):
        raise ConnectionError("upload failed")

    with monkeypatch.context() as m:
        m.setattr(ConnectionRouter, "owner_write_event_messages_to_outbox", fail_upload)
        do_manager.sync(auto_checkpoint=False)

    # The failed messages must not go out after the revoke.
    folder.revoke_read_access(ds_manager.email)
    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager) == []


def test_read_error_leaves_no_planned_message(monkeypatch):
    ds_manager, do_manager = _pair()
    syncer = do_manager.datasite_owner_syncer
    shared = syncer.perm_context.open("shared/")
    shared.grant_read_access(ds_manager.email)
    shared.grant_read_access("peer_b@example.com")
    _write(do_manager, "shared/data.txt", "one")
    syncer._record_local_changes()

    # The plan reads the file once per peer. The read for the second peer fails.
    reads = []
    create_event = (
        datasite_owner_syncer.DatasiteOwnerSyncer._create_current_filestate_event
    )

    def fail_second_read(self, path):
        reads.append(path)
        if len(reads) == 2:
            raise PermissionError(path)
        return create_event(self, path)

    monkeypatch.setattr(
        datasite_owner_syncer.DatasiteOwnerSyncer,
        "_create_current_filestate_event",
        fail_second_read,
    )
    with pytest.raises(PermissionError):
        syncer.deliver_file_changes([ds_manager.email, "peer_b@example.com"])
    assert syncer.outbox_queue.empty()


def test_compaction_keeps_message_size_cap(monkeypatch):
    monkeypatch.setattr(datasite_owner_syncer, "MAX_DELIVERY_MESSAGE_CONTENT", 3)
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    for name in ("a", "b", "c", "d"):
        _write(do_manager, f"shared/{name}.txt", "abc")
    do_manager.sync(auto_checkpoint=False)

    monkeypatch.setattr(datasite_owner_syncer, "MAX_DELIVERY_MESSAGE_CONTENT", 6)
    syncer = do_manager.datasite_owner_syncer
    assert syncer.compact_outbox_if_needed(ds_manager.email, min_messages=2) == 4
    messages = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    assert [len(m.events) for m in messages] == [2, 2]
