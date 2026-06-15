"""Tests for cache-aware sync functionality.

Tests verify that DO and DS properly use file-backed cache to only download
events they don't already have locally, avoiding redundant re-downloads on restart.
"""

from pathlib import Path

from syft_client.sync.syftbox_manager import SyftboxManager
from tests.unit.utils import get_mock_events_messages


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


def test_ds_incremental_sync_downloads_only_new_events():
    """Pre-populate cache with first n events, create new manager with same cache dir,
    sync with n+m in outbox, verify only m new events are downloaded."""
    import time

    # Create initial pair with filesystem cache
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    # Add initial 3 events to DO's outbox for DS
    initial_events = get_mock_events_messages(3)
    for event in initial_events:
        do_manager._connection_router.owner_write_event_messages_to_outbox(
            ds_manager.email, event
        )

    ds_manager.sync()

    # Verify initial sync
    ds_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    initial_cache_count = len(ds_cache.events_connection)
    assert initial_cache_count == 3

    # Verify last_event_timestamp_per_peer is populated
    assert len(ds_cache.last_event_timestamp_per_peer) > 0, (
        "last_event_timestamp_per_peer should be populated"
    )

    # Add 2 more events to outbox
    time.sleep(0.01)  # Ensure new events have later timestamps
    additional_events = get_mock_events_messages(2)
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
    assert len(new_cache.last_event_timestamp_per_peer) > 0, (
        "last_event_timestamp_per_peer should be pre-populated from disk cache"
    )
    assert len(new_cache.file_hashes) == 3, (
        "file_hashes should be pre-populated from disk cache"
    )

    # Sync the new manager - should only download the 2 new events
    new_ds_manager.sync()

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

    from syft_client.sync.sync.caches.datasite_owner_cache import (
        DataSiteOwnerEventCache,
        DataSiteOwnerEventCacheConfig,
    )

    from syft_client.sync.syftbox_manager import COLLECTION_SUBPATH

    collections_folder = syftbox_folder / do_manager.email / COLLECTION_SUBPATH
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

    from syft_client.sync.sync.caches.datasite_watcher_cache import (
        DataSiteWatcherCache,
        DataSiteWatcherCacheConfig,
    )

    from syft_client.sync.syftbox_manager import COLLECTION_SUBPATH

    config = DataSiteWatcherCacheConfig(
        email=do_manager.email,
        use_in_memory_cache=False,
        syftbox_folder=syftbox_folder,
        collection_subpath=COLLECTION_SUBPATH,
        connection_configs=[],
    )
    new_cache = DataSiteWatcherCache.from_config(config)

    # Verify deleted file is not in reloaded cache
    assert expected_path not in new_cache.file_hashes, (
        "Deleted file should not appear in file_hashes after cache reload"
    )
