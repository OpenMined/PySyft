"""The data owner sends each peer what it lacks, and deletes what it may no
longer have, from a per-peer record of what the peer holds."""

import threading

import pytest

from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.sync.constants import CACHE_DIR
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

PEER = "ds@example.com"


def _tracker_holding(view: dict[str, str]) -> DeliveryTracker:
    tracker = DeliveryTracker(MemoryDeliveryStateStore())
    tracker.sent(PEER, "m0", SendStatus.DELIVERED, view)
    return tracker


def test_plan_sends_changed_files_deletes_unreadable():
    tracker = _tracker_holding({"same": "h1", "old": "h1", "revoked": "h1"})
    plan = tracker.plan({PEER: {"same": "h1", "old": "h2", "new": "h3"}})[PEER]
    assert plan.send == ["new", "old"]
    assert plan.delete == ["revoked"]


def test_peer_without_view_gets_all_readable_files():
    tracker = DeliveryTracker(MemoryDeliveryStateStore())
    assert tracker.plan({PEER: {"a": "h1", "b": "h2"}})[PEER].send == ["a", "b"]


def test_delivered_send_updates_view():
    tracker = _tracker_holding({"a": "h1", "b": "h1"})
    tracker.sent(PEER, "m1", SendStatus.DELIVERED, {"a": "h2", "b": None})
    assert tracker.plan({PEER: {"a": "h2"}})[PEER].send == []
    assert tracker.plan({PEER: {"a": "h2"}})[PEER].delete == []


def test_queued_send_updates_view_on_confirm():
    tracker = DeliveryTracker(MemoryDeliveryStateStore())
    tracker.sent(PEER, "m1", SendStatus.QUEUED, {"a": "h1"})
    assert tracker.plan({PEER: {"a": "h1"}})[PEER].send == ["a"]
    tracker.confirm(["m1"])
    assert tracker.plan({PEER: {"a": "h1"}})[PEER].send == []


def test_local_store_persists_state_refuses_later_version(tmp_path):
    path = tmp_path / DELIVERY_VIEWS_FILENAME
    DeliveryTracker(LocalDeliveryStateStore(path)).sent(
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
        tracker.sent(peer, f"m-{peer}", SendStatus.DELIVERED, {"a": "h1"})

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


def test_in_memory_owner_sends_file_content():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    syncer = do_manager.datasite_owner_syncer
    syncer.perm_context.open("shared/").grant_read_access(ds_manager.email)
    syncer.event_cache.file_connection.write_file("shared/data.txt", "hello")

    do_manager.sync(auto_checkpoint=False)
    do_manager.sync(auto_checkpoint=False)
    messages = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    sent = [
        (e.is_deleted, e.content)
        for m in messages
        for e in m.events
        if str(e.path_in_datasite) == "shared/data.txt"
    ]
    assert sent == [(False, "hello")]


@pytest.mark.parametrize("persist_owner_state", [True, False])
def test_revoked_peer_gets_delete(persist_owner_state):
    ds_manager, do_manager = _pair(persist_owner_state=persist_owner_state)
    folder = do_manager.datasite_owner_syncer.perm_context.open("shared/")
    folder.grant_read_access(ds_manager.email)
    _write(do_manager, "shared/data.txt", "secret")
    do_manager.sync(auto_checkpoint=False)
    ds_manager.sync()
    received = _received(ds_manager, do_manager, "shared/data.txt")
    assert received.exists()

    folder.revoke_read_access(ds_manager.email)
    do_manager.sync(auto_checkpoint=False)
    ds_manager.sync()
    assert not received.exists(), "The DS kept a file it may no longer read."


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


def test_lost_views_resend_everything():
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/data.txt", "one")
    do_manager.sync(auto_checkpoint=False)
    (do_manager.syftbox_folder / CACHE_DIR / DELIVERY_VIEWS_FILENAME).unlink()

    do_manager.sync(auto_checkpoint=False)
    assert _outbox_paths(ds_manager, do_manager).count("shared/data.txt") == 2


def test_failed_outbox_write_resent_next_sync(monkeypatch):
    ds_manager, do_manager = _pair()
    do_manager.datasite_owner_syncer.perm_context.open("shared/").grant_read_access(
        ds_manager.email
    )
    _write(do_manager, "shared/data.txt", "one")

    def fail_upload(*args, **kwargs):
        raise ConnectionError("upload failed")

    with monkeypatch.context() as m:
        m.setattr(ConnectionRouter, "owner_write_event_messages_to_outbox", fail_upload)
        with pytest.raises(ConnectionError):
            do_manager.sync(auto_checkpoint=False)

    do_manager.sync(auto_checkpoint=False)
    ds_manager.sync()
    assert _received(ds_manager, do_manager, "shared/data.txt").read_text() == "one"
