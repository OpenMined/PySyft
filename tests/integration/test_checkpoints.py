"""
Integration tests for checkpoint functionality with Google Drive.

Tests GDrive-specific checkpoint behavior that can't be tested with in-memory connections.
Core checkpoint logic (create, restore, threshold, deduplication, compacting) is
covered by unit tests.

These tests validate:
1. Full checkpoint + incremental + rolling state restoration on real GDrive
2. Auto-checkpoint creation + compacting + fresh login restore on real GDrive
"""

import os
from time import sleep

import pytest

from syft_client.sync.syftbox_manager import SyftboxManager
from tests.integration.utils import token_path_do, token_path_ds


EMAIL_DO = os.environ.get("AI_AUDIT_EMAIL_DO", "")
EMAIL_DS = os.environ.get("AI_AUDIT_EMAIL_DS", "")


@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_checkpoint_with_incremental_events():
    """Test that full checkpoint + incremental checkpoints + rolling state works correctly."""
    # First session: create initial state and full checkpoint
    manager_ds1, manager_do1 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
        )
    )

    manager_do1.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        manager_ds1.email
    )

    # Send initial files
    manager_ds1._send_file_change(f"{EMAIL_DO}/initial1.txt", "Initial 1")
    manager_ds1._send_file_change(f"{EMAIL_DO}/initial2.txt", "Initial 2")
    sleep(1)

    manager_do1.sync(auto_checkpoint=False)

    # Create full checkpoint (legacy behavior)
    full_checkpoint = manager_do1.create_checkpoint()
    assert len(full_checkpoint.files) == 2
    checkpoint_timestamp = full_checkpoint.last_event_timestamp
    print(f"Full checkpoint created with last_event_timestamp: {checkpoint_timestamp}")

    # Send more files to create incremental checkpoint
    manager_ds1._send_file_change(f"{EMAIL_DO}/inc1.txt", "Incremental 1")
    manager_ds1._send_file_change(f"{EMAIL_DO}/inc2.txt", "Incremental 2")
    sleep(1)

    manager_do1.sync(auto_checkpoint=False)

    # Create incremental checkpoint (new flow)
    manager_do1.try_create_checkpoint(threshold=2)

    # Verify incremental checkpoint was created
    inc_cps = manager_do1._connection_router.get_all_incremental_checkpoints()
    assert len(inc_cps) >= 1, "Should have at least 1 incremental checkpoint"
    print(f"Created {len(inc_cps)} incremental checkpoint(s)")

    # Send more files for rolling state
    manager_ds1._send_file_change(f"{EMAIL_DO}/rolling1.txt", "Rolling 1")
    sleep(1)

    manager_do1.sync(auto_checkpoint=False)

    # Verify DO1 has all 5 files
    assert len(manager_do1.datasite_owner_syncer.event_cache.file_hashes) == 5

    # Fresh login should restore: full checkpoint + incremental checkpoints + rolling state
    manager_ds2, manager_do2 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            add_peers=False,
            load_peers=True,
            clear_caches=True,
        )
    )

    manager_do2.sync(auto_checkpoint=False)

    # Should have all 5 files (2 from full checkpoint + 2 from incremental + 1 from rolling state)
    do_cache = manager_do2.datasite_owner_syncer.event_cache
    assert len(do_cache.file_hashes) == 5, (
        f"Expected 5 files, got {len(do_cache.file_hashes)}"
    )
    print(
        f"Restored {len(do_cache.file_hashes)} files (full checkpoint + incremental + rolling state)"
    )


@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_auto_checkpoint_on_sync():
    """Test automatic incremental checkpoint creation and compacting during sync."""
    manager_ds, manager_do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
    )

    manager_do.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        manager_ds.email
    )

    # Send enough files to trigger auto-incremental-checkpoint (threshold=5)
    for i in range(6):
        manager_ds._send_file_change(f"{EMAIL_DO}/auto{i}.txt", f"Auto content {i}")
    sleep(1)

    # Sync with auto_checkpoint enabled and low threshold
    manager_do.sync(auto_checkpoint=True, checkpoint_threshold=5)

    # Verify incremental checkpoint was created
    inc_cps = manager_do._connection_router.get_all_incremental_checkpoints()
    assert len(inc_cps) >= 1, "Auto-incremental-checkpoint should have been created"
    print(f"Auto-created {len(inc_cps)} incremental checkpoint(s)")

    # Test compacting: Create more incremental checkpoints
    for i in range(6, 12):
        manager_ds._send_file_change(f"{EMAIL_DO}/auto{i}.txt", f"Auto content {i}")
    sleep(1)

    manager_do.sync(auto_checkpoint=False)
    manager_do.try_create_checkpoint(threshold=5)  # Create 2nd incremental checkpoint

    # Manually compact to test merging
    compacted = manager_do.datasite_owner_syncer.compact_checkpoints()
    assert len(compacted.files) == 12, (
        f"Expected 12 files after compacting, got {len(compacted.files)}"
    )
    print(f"Compacted checkpoint has {len(compacted.files)} files")

    # Verify incremental checkpoints were deleted after compacting
    inc_cps_after = manager_do._connection_router.get_all_incremental_checkpoints()
    assert len(inc_cps_after) == 0, (
        "Incremental checkpoints should be deleted after compacting"
    )

    # Fresh login should restore from compacted checkpoint
    manager_ds2, manager_do2 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            add_peers=False,
            load_peers=True,
            clear_caches=True,
        )
    )

    manager_do2.sync(auto_checkpoint=False)

    # Should have all 12 files from compacted checkpoint
    do_cache = manager_do2.datasite_owner_syncer.event_cache
    assert len(do_cache.file_hashes) == 12, (
        f"Expected 12 files from compacted checkpoint, got {len(do_cache.file_hashes)}"
    )
    print(f"Restored {len(do_cache.file_hashes)} files from compacted checkpoint")
