"""Integration tests for cache-aware sync functionality with Google Drive.

These tests require valid Google Drive credentials and verify that the cache-aware
sync works correctly with real GDrive operations.
"""

import os
from pathlib import Path
from time import sleep

import pytest

from syft_client.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from tests.integration.with_unit_coverage.utils import is_mock_mode


SYFT_CLIENT_DIR = Path(__file__).parent.parent.parent.parent
CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"

FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ.get("AI_AUDIT_EMAIL_DO", "do@test.com")

FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ.get("AI_AUDIT_EMAIL_DS", "ds@test.com")

token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS


@pytest.mark.skipif(is_mock_mode(), reason="Cache restart tests require real GDrive")
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_do_cache_aware_sync_gdrive():
    """Real GDrive test: Create DO, sync 3 files, create 2 more, restart with same cache,
    verify cache has 3 events, sync and end up with 5.

    This test verifies that when a DO manager is restarted with a persisted filesystem cache,
    it only downloads events newer than what's already cached, reducing network calls.
    """
    # Create initial pair with filesystem cache
    ds_manager1, do_manager1 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            use_in_memory_cache=False,
        )
    )

    # DS sends 3 files to DO
    sleep(1)
    ds_manager1._send_file_change(
        f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/file1.job", "Content 1"
    )
    ds_manager1._send_file_change(
        f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/file2.job", "Content 2"
    )
    ds_manager1._send_file_change(
        f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/file3.job", "Content 3"
    )
    sleep(1)

    # DO syncs to receive and cache the files
    do_manager1.sync()

    # Verify initial cache state
    do_cache1 = do_manager1.datasite_owner_syncer.event_cache
    initial_cache_count = len(do_cache1.events_messages_connection)
    initial_hash_count = len(do_cache1.file_hashes)
    assert initial_cache_count >= 3, (
        f"Should have at least 3 event messages cached, got {initial_cache_count}"
    )
    assert initial_hash_count >= 3, (
        f"Should have at least 3 file hashes cached, got {initial_hash_count}"
    )

    # DS sends 2 more files
    sleep(1)
    ds_manager1._send_file_change(
        f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/file4.job", "Content 4"
    )
    ds_manager1._send_file_change(
        f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/file5.job", "Content 5"
    )
    sleep(1)

    local_syftbox_folder_do = do_manager1.syftbox_folder
    # Create a NEW DO manager with the same cache directory (simulating restart)
    new_do_config = SyftboxManagerConfig.for_google_drive_testing_connection(
        email=EMAIL_DO,
        token_path=token_path_do,
        syftbox_folder=local_syftbox_folder_do,
        has_ds_role=False,
        has_do_role=True,
        use_in_memory_cache=False,
        check_versions=False,
    )
    new_do_manager = SyftboxManager.from_config(new_do_config)

    # Verify the new manager's cache loaded existing events from disk
    new_cache = new_do_manager.datasite_owner_syncer.event_cache
    assert len(new_cache.events_messages_connection) == initial_cache_count, (
        f"New manager cache should have loaded {initial_cache_count} events from disk, "
        f"got {len(new_cache.events_messages_connection)}"
    )
    assert new_cache.latest_cached_timestamp is not None, (
        "New manager cache should have latest_cached_timestamp set"
    )
    assert len(new_cache.file_hashes) >= 3, (
        f"New manager should have at least 3 file hashes pre-populated, got {len(new_cache.file_hashes)}"
    )

    # Sync the new manager - should only download the 2 new events
    new_do_manager.sync()

    # Verify total events in cache (should be initial + 2 new)
    final_cache_count = len(new_cache.events_messages_connection)
    assert final_cache_count >= initial_cache_count + 2, (
        f"Should have at least {initial_cache_count + 2} events after sync, got {final_cache_count}"
    )


@pytest.mark.skipif(is_mock_mode(), reason="Cache restart tests require real GDrive")
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_ds_cache_aware_sync_gdrive():
    """Real GDrive test: Create DS, sync 3 files, create 2 more, restart with same cache,
    verify cache has 3 events, sync and end up with 5.

    This test verifies that when a DS manager is restarted with a persisted filesystem cache,
    it uses the cached last_event_timestamp_per_peer to only download new events.
    """
    # Create initial pair with filesystem cache
    ds_manager1, do_manager1 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            use_in_memory_cache=False,
        )
    )

    do_manager1.datasite_owner_syncer.perm_context.open(".").grant_read_access(
        ds_manager1.email
    )

    # DO creates 3 files
    do_datasite_dir = do_manager1.syftbox_folder / do_manager1.email
    do_datasite_dir.mkdir(parents=True, exist_ok=True)

    (do_datasite_dir / "file1.txt").write_text("Content 1")
    (do_datasite_dir / "file2.txt").write_text("Content 2")
    (do_datasite_dir / "file3.txt").write_text("Content 3")

    # DO syncs to propagate files
    do_manager1.sync()
    sleep(1)

    # DS syncs to receive files
    ds_manager1.sync()

    # Verify initial cache state
    ds_cache1 = ds_manager1.datasite_watcher_syncer.datasite_watcher_cache
    initial_cache_count = len(ds_cache1.events_connection)
    initial_hash_count = len(ds_cache1.file_hashes)
    assert initial_cache_count >= 1, (
        f"Should have at least 1 event message cached, got {initial_cache_count}"
    )
    assert initial_hash_count >= 3, (
        f"Should have at least 3 file hashes cached, got {initial_hash_count}"
    )
    assert len(ds_cache1.last_event_timestamp_per_peer) > 0, (
        "Should have timestamps for peers"
    )

    # Save the syftbox_folder for creating a new manager
    syftbox_folder = ds_manager1.syftbox_folder

    # DO creates 2 more files
    sleep(1)
    (do_datasite_dir / "file4.txt").write_text("Content 4")
    (do_datasite_dir / "file5.txt").write_text("Content 5")
    do_manager1.sync()
    sleep(1)

    # Create a NEW DS manager with the same cache directory (simulating restart)
    new_ds_config = SyftboxManagerConfig.for_google_drive_testing_connection(
        email=EMAIL_DS,
        token_path=token_path_ds,
        syftbox_folder=syftbox_folder,
        has_ds_role=True,
        has_do_role=False,
        use_in_memory_cache=False,
        check_versions=False,
    )
    new_ds_manager = SyftboxManager.from_config(new_ds_config)

    # Verify the new manager's cache loaded existing events from disk
    new_cache = new_ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert len(new_cache.events_connection) == initial_cache_count, (
        f"New manager cache should have loaded {initial_cache_count} events from disk, "
        f"got {len(new_cache.events_connection)}"
    )
    assert len(new_cache.last_event_timestamp_per_peer) > 0, (
        "last_event_timestamp_per_peer should be pre-populated from disk cache"
    )
    assert len(new_cache.file_hashes) >= 3, (
        f"New manager should have at least 3 file hashes pre-populated, got {len(new_cache.file_hashes)}"
    )

    # Sync the new manager - should only download the new events
    new_ds_manager.sync()

    # Verify file hashes increased (should have 5 files total now)
    final_hash_count = len(new_cache.file_hashes)
    assert final_hash_count >= 5, (
        f"Should have at least 5 file hashes after sync, got {final_hash_count}"
    )
