"""Tests for file-backed cache persistence.

If the cache persistence feature is removed or broken, these tests fail.
"""

from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.sync.constants import (
    CACHE_DIR,
    OWNER_FILE_HASHES_FILENAME,
    ROLLING_STATE_FILENAME,
)


def test_no_duplicate_events_across_processes():
    """Two managers sharing a syftbox_folder must not create duplicate events
    for the same file change.

    Without caching: Process B's file_hashes is stale → re-detects Process A's
    changes → uploads duplicate events to GDrive → peers receive them twice.

    With caching: Process B loads A's file_hashes → sees the file was already
    handled → no duplicate event.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    # Both a local DO change and a peer DS change
    local_file = do_manager.syftbox_folder / do_manager.email / "local.txt"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text("DO wrote this")
    ds_manager._send_file_change(f"{do_manager.email}/from_peer.txt", "DS wrote this")

    # Process A syncs — detects changes, uploads events, saves cache
    do_manager.sync(auto_checkpoint=False)
    events_after_a = len(
        do_manager.datasite_owner_syncer.connection_router.owner_get_all_accepted_event_file_ids()
    )
    assert events_after_a > 0

    # Process B (syft-bg) syncs — must NOT re-create the same events
    do_manager_b = do_manager._copy()
    do_manager_b.sync(auto_checkpoint=False)
    events_after_b = len(
        do_manager_b.datasite_owner_syncer.connection_router.owner_get_all_accepted_event_file_ids()
    )

    # Verify cache files were created
    cache_dir = do_manager.syftbox_folder / CACHE_DIR
    assert (cache_dir / OWNER_FILE_HASHES_FILENAME).exists(), (
        f"{OWNER_FILE_HASHES_FILENAME} not created after sync"
    )
    assert (cache_dir / ROLLING_STATE_FILENAME).exists(), (
        f"{ROLLING_STATE_FILENAME} not created after sync"
    )

    assert events_after_b == events_after_a, (
        f"Duplicate events! After A: {events_after_a}, after B: {events_after_b}. "
        f"Process B re-detected changes that A already handled."
    )


def test_rolling_state_not_overwritten_across_processes():
    """Simulates the real scenario: syft-bg has been running for a while,
    then the notebook syncs new events. syft-bg syncs again — its rolling
    state must include the notebook's events, not overwrite them.

    Without caching: B's in-memory rolling state is stale (from its initial
    sync hours ago). When B syncs again, it uploads its stale state to GDrive,
    erasing A's new events. Data loss for new clients.

    With caching: B loads A's saved rolling state from .cache/ → builds on
    top of it → uploads the merged state → nothing lost.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    # --- Step 1: Process B (syft-bg) starts first and does its initial sync ---
    do_manager_b = do_manager._copy()
    do_manager_b.sync(auto_checkpoint=False)
    # B is now "running" — initial_sync_done = True, has its own rolling state

    # --- Step 2: Process A (notebook) gets peer events and syncs ---
    ds_manager._send_file_change(f"{do_manager.email}/file1.txt", "content1")
    ds_manager._send_file_change(f"{do_manager.email}/file2.txt", "content2")
    do_manager.sync(auto_checkpoint=False)

    # A's rolling state now has file1 and file2
    rs_after_a = do_manager.datasite_owner_syncer._rolling_state
    assert rs_after_a is not None
    assert rs_after_a.event_count >= 2

    # --- Step 3: Process B (syft-bg) syncs again ---
    # In real life: B has been running, initial_sync_done=True,
    # pull_initial_state is SKIPPED. B's rolling state is stale.
    ds_manager._send_file_change(f"{do_manager.email}/file3.txt", "content3")
    do_manager_b.sync(auto_checkpoint=False)

    # B's rolling state must include A's events (file1, file2) + its own (file3)
    rs_after_b = do_manager_b.datasite_owner_syncer._rolling_state
    assert rs_after_b is not None

    paths_in_rs = {str(e.path_in_datasite) for e in rs_after_b.events}
    assert "file1.txt" in paths_in_rs, (
        "file1.txt lost! B overwrote A's rolling state instead of building on it."
    )
    assert "file2.txt" in paths_in_rs, (
        "file2.txt lost! B overwrote A's rolling state instead of building on it."
    )
    assert "file3.txt" in paths_in_rs, "file3.txt missing from rolling state."
