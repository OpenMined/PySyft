"""Tests for cache-aware sync functionality.

Tests verify that DO and DS properly use file-backed cache to only download
events they don't already have locally, avoiding redundant re-downloads on restart.
"""

import warnings
from pathlib import Path

import pytest

from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.connections.types import MessageKind
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.sync.collection_spec import CollectionSyncSpec

from tests.unit.utils import (
    router_with_mock_connection,
    TEST_COLLECTION_PREFIX,
    TEST_COLLECTION_SUBPATH,
    get_mock_events_messages,
)


def test_do_incremental_sync_downloads_only_new_events():
    """Pre-populate cache with first n events, create new manager with same cache dir,
    sync with n+m in backend, verify only m new events are downloaded."""

    # Create initial pair with filesystem cache
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    # Add initial 3 events to the personal SyftBox folder
    initial_events = get_mock_events_messages(3)
    for event in initial_events:
        do_manager._connection_router.owner_write_events_message_to_syftbox(event)

    do_manager.sync()

    # Verify initial sync
    initial_cache_count = len(
        do_manager.datasite_owner_syncer.event_cache.events_messages_connection
    )
    assert initial_cache_count >= 3  # At least 3 from backend

    # Add 2 more events to backend
    additional_events = get_mock_events_messages(2)
    for event in additional_events:
        do_manager._connection_router.owner_write_events_message_to_syftbox(event)

    # Create a NEW SyftboxManager with the same cache directory (simulating restart)
    new_do_manager = do_manager._copy()

    # Verify the new manager's cache loaded existing events from disk
    new_cache = new_do_manager.datasite_owner_syncer.event_cache
    assert len(new_cache.events_messages_connection) == initial_cache_count, (
        f"New manager cache should have loaded {initial_cache_count} events from disk"
    )
    assert new_cache.latest_cached_timestamp is not None, (
        "New manager cache should have latest_cached_timestamp set"
    )

    # Track download calls on the new manager
    download_call_count = 0
    original_download = new_do_manager.datasite_owner_syncer.download_events_message_by_id_with_connection

    def counted_download(events_message_id):
        nonlocal download_call_count
        download_call_count += 1
        return original_download(events_message_id)

    new_do_manager.datasite_owner_syncer.download_events_message_by_id_with_connection = counted_download

    # Sync the new manager (with recompute_hashes=False to avoid additional local events)
    new_do_manager.datasite_owner_syncer.sync(
        peer_emails=[ds_manager.email], recompute_hashes=False
    )

    assert download_call_count <= 3, (
        f"Should download only new events, not all of them. "
        f"Got {download_call_count} downloads."
    )

    # Verify we now have 2 more events than before
    final_cache_count = len(new_cache.events_messages_connection)
    assert final_cache_count == initial_cache_count + 2, (
        f"Should have {initial_cache_count + 2} events, got {final_cache_count}"
    )


def test_ds_incremental_sync_downloads_only_new_events(monkeypatch):
    """Pre-populate cache with first n events, create new manager with same cache dir,
    sync with n+m in outbox, verify only m new events are downloaded."""
    import time

    # Create initial pair with filesystem cache
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    # Add initial 3 events to DO's outbox for DS
    initial_events = get_mock_events_messages(3, datasite_email=do_manager.email)
    for event in initial_events:
        do_manager._connection_router.owner_write_event_messages_to_outbox(
            ds_manager.email, event
        )

    ds_manager.sync()

    # Verify initial sync
    ds_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    initial_cache_count = len(ds_cache.events_connection)
    assert initial_cache_count == 3

    assert do_manager.email in ds_cache.last_msg_id_per_peer

    # Add 2 more events to outbox
    time.sleep(0.01)  # Ensure new events have later timestamps
    additional_events = get_mock_events_messages(2, datasite_email=do_manager.email)
    for event in additional_events:
        do_manager._connection_router.owner_write_event_messages_to_outbox(
            ds_manager.email, event
        )

    # Create a NEW SyftboxManager with the same cache directory (simulating restart)
    new_ds_manager = ds_manager._copy()

    # Verify the new manager's cache loaded existing events from disk
    new_cache = new_ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert len(new_cache.events_connection) == initial_cache_count, (
        f"New manager cache should have loaded {initial_cache_count} events from disk"
    )
    assert (
        new_cache.last_msg_id_per_peer[do_manager.email]
        == ds_cache.last_msg_id_per_peer[do_manager.email]
    ), "last_msg_id_per_peer should be pre-populated from disk cache"
    assert len(new_cache.file_hashes) == 3, (
        "file_hashes should be pre-populated from disk cache"
    )

    # Sync the new manager - should only download the 2 new events
    p2p = new_ds_manager._connection_router.connections.p2p_con
    original_receive = p2p.receive
    received = []

    def counted_receive(*args, **kwargs):
        messages = original_receive(*args, **kwargs)
        received.extend(m for m in messages if m.kind is MessageKind.FILE_EVENTS)
        return messages

    monkeypatch.setattr(p2p, "receive", counted_receive)
    new_ds_manager.sync()
    assert len(received) == 2, (
        f"Should receive only 2 new messages, got {len(received)}"
    )

    # Verify total events in cache
    assert len(new_cache.events_connection) == 5, (
        "Should have 5 total events after syncing new events"
    )


def test_do_cache_handles_deletions_correctly():
    """Test that file_hashes properly reflects deletions when loading from cache."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email

    # Create a file
    test_file = datasite_dir_do / "test_file.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("Hello")

    do_manager.sync()

    # Verify file is in cache
    do_cache = do_manager.datasite_owner_syncer.event_cache
    assert "test_file.txt" in [str(p) for p in do_cache.file_hashes.keys()]

    # Delete the file
    test_file.unlink()

    do_manager.sync()

    # Verify file is removed from cache
    assert "test_file.txt" not in [str(p) for p in do_cache.file_hashes.keys()]

    # Now test cache reload - deletion should still be reflected
    syftbox_folder = do_manager.syftbox_folder

    from syft.sync.sync.caches.datasite_owner_cache import (
        DataSiteOwnerEventCache,
        DataSiteOwnerEventCacheConfig,
    )

    collections_folder = syftbox_folder / do_manager.email / TEST_COLLECTION_SUBPATH
    config = DataSiteOwnerEventCacheConfig(
        use_in_memory_cache=False,
        syftbox_folder=syftbox_folder,
        email=do_manager.email,
        collections_folder=collections_folder,
    )
    new_cache = DataSiteOwnerEventCache.from_config(config)

    # Verify deleted file is not in reloaded cache's file_hashes
    assert "test_file.txt" not in [str(p) for p in new_cache.file_hashes.keys()], (
        "Deleted file should not appear in file_hashes after cache reload"
    )


def test_ds_cache_handles_deletions_correctly():
    """Test that DS file_hashes properly reflects deletions when loading from cache."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email

    # Grant DS read access so files sync to DS
    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open(".").grant_read_access(ds_manager.email)

    # Create a file on DO side
    test_file = datasite_dir_do / "test_file.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("Hello")

    do_manager.sync()
    ds_manager.sync()

    ds_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache

    # Verify file is in DS cache
    expected_path = Path(do_manager.email) / "test_file.txt"
    assert expected_path in ds_cache.file_hashes

    # Delete the file on DO side
    test_file.unlink()

    do_manager.sync()
    ds_manager.sync()

    # Verify file is removed from DS cache
    assert expected_path not in ds_cache.file_hashes

    # Test cache reload
    syftbox_folder = ds_manager.syftbox_folder

    from syft.sync.sync.caches.datasite_watcher_cache import (
        DataSiteWatcherCache,
        DataSiteWatcherCacheConfig,
    )

    config = DataSiteWatcherCacheConfig(
        email=do_manager.email,
        use_in_memory_cache=False,
        syftbox_folder=syftbox_folder,
        collection_specs=[
            CollectionSyncSpec.public(TEST_COLLECTION_PREFIX, TEST_COLLECTION_SUBPATH)
        ],
    )
    new_cache = DataSiteWatcherCache.from_config(
        config, router_with_mock_connection(do_manager.email)
    )

    # Verify deleted file is not in reloaded cache
    assert expected_path not in new_cache.file_hashes, (
        "Deleted file should not appear in file_hashes after cache reload"
    )


# -- reading the DO's outbox -------------------------------------------------------


def _pair():
    return SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )


def _msg_id(message: FileChangeEventsMessage) -> str:
    return message.message_filepath.as_string()


def _sent_at(message: FileChangeEventsMessage, timestamp: float):
    """``message`` under a name that carries ``timestamp``."""
    name = FileChangeEventsMessageFileName(timestamp=timestamp)
    return message.model_copy(update={"message_filepath": name})


def _applied_msg_ids(ds_manager) -> set[str]:
    cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    return {_msg_id(m) for m in cache.events_connection.get_all()}


def test_ds_receives_message_landing_after_newer_one():
    ds_manager, do_manager = _pair()
    late, newer = get_mock_events_messages(2, datasite_email=do_manager.email)
    late = _sent_at(late, newer.timestamp - 10)
    router = do_manager._connection_router

    router.owner_write_event_messages_to_outbox(ds_manager.email, newer)
    ds_manager.sync()
    router.owner_write_event_messages_to_outbox(ds_manager.email, late)
    ds_manager.sync()

    cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert _applied_msg_ids(ds_manager) == {_msg_id(late), _msg_id(newer)}
    assert cache.last_msg_id_per_peer[do_manager.email] == _msg_id(newer)


def test_ds_sync_waits_for_message_that_failed_to_download(monkeypatch):
    ds_manager, do_manager = _pair()
    first, second = get_mock_events_messages(2, datasite_email=do_manager.email)
    second = _sent_at(second, first.timestamp + 1)
    router = do_manager._connection_router
    for message in (first, second):
        router.owner_write_event_messages_to_outbox(ds_manager.email, message)
    [first_id] = [
        f["id"]
        for f in router.owner_get_outbox_file_metadatas(ds_manager.email)
        if f["name"] == _msg_id(first)
    ]
    original_download = GDriveClient.download_file

    def fail_first(self, file_id: str) -> bytes:
        if file_id == first_id:
            raise ValueError("download cut off")
        return original_download(self, file_id)

    with monkeypatch.context() as m:
        m.setattr(GDriveClient, "download_file", fail_first)
        with pytest.warns(UserWarning, match="download failed"):
            ds_manager.sync()
    assert _applied_msg_ids(ds_manager) == set()

    ds_manager.sync()
    assert _applied_msg_ids(ds_manager) == {_msg_id(first), _msg_id(second)}


def test_ds_sync_passes_over_unreadable_message():
    ds_manager, do_manager = _pair()
    [readable] = get_mock_events_messages(1, datasite_email=do_manager.email)
    unreadable = FileChangeEventsMessageFileName(timestamp=readable.timestamp - 1)
    router = do_manager._connection_router
    router.connections.p2p_con.send(
        ds_manager.email, unreadable.as_string(), b"garbage", MessageKind.FILE_EVENTS
    )
    router.owner_write_event_messages_to_outbox(ds_manager.email, readable)

    with pytest.warns(UserWarning, match="Skipped unreadable outbox message"):
        ds_manager.sync()
    assert _applied_msg_ids(ds_manager) == {_msg_id(readable)}

    # Received once: the next sync does not download it again.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds_manager.sync()
    assert not [w for w in caught if unreadable.as_string() in str(w.message)]


def test_restored_bookmark_found_for_peer_email_in_other_case(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    [message] = get_mock_events_messages(1, datasite_email=do_manager.email)
    do_manager._connection_router.owner_write_event_messages_to_outbox(
        ds_manager.email, message
    )
    ds_manager.sync()

    cache = ds_manager._copy().datasite_watcher_syncer.datasite_watcher_cache
    p2p = cache.connection_router.connections.p2p_con
    original_receive = p2p.receive
    afters = []

    def capture_after(*args, **kwargs):
        afters.append(kwargs.get("after"))
        return original_receive(*args, **kwargs)

    monkeypatch.setattr(p2p, "receive", capture_after)
    cache.sync_down(do_manager.email.upper())
    assert afters == [_msg_id(message)]
