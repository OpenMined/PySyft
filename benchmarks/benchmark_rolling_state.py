"""Benchmark rolling state optimization for fresh login sync.

This benchmark tests the rolling state feature which accumulates events after
a checkpoint, allowing fresh logins to sync with minimal API calls:
- Without rolling state: 1 checkpoint download + N individual event downloads
- With rolling state: 1 checkpoint download + 1 rolling state download

Test scenario (two-phase approach to ensure rolling state is populated):
- PHASE 1: Create EVENTS_BEFORE_CHECKPOINT events, sync DO → creates checkpoint
           (rolling state is initialized when checkpoint is created)
- PHASE 2: Create EVENTS_AFTER_CHECKPOINT events, sync DO → events go into rolling state
           (these events are tracked because rolling state now exists)
- Fresh login should verify rolling state contains EVENTS_AFTER_CHECKPOINT events
"""

import os
import time
from pathlib import Path
from syft_client.sync.syftbox_manager import SyftboxManager

REPO_ROOT = Path(__file__).parent.parent
CREDENTIALS_DIR = REPO_ROOT / "credentials"
EMAIL_DO = os.environ["AI_AUDIT_EMAIL_DO"]
EMAIL_DS = os.environ["AI_AUDIT_EMAIL_DS"]
token_path_do = CREDENTIALS_DIR / os.environ.get(
    "ai_audit_credentials_fname_do", "token_do.json"
)
token_path_ds = CREDENTIALS_DIR / os.environ.get(
    "ai_audit_credentials_fname_ds", "token_ds.json"
)

EVENTS_BEFORE_CHECKPOINT = 3  # Events to create before checkpoint (triggers checkpoint)
EVENTS_AFTER_CHECKPOINT = (
    2  # Events to create after checkpoint (goes into rolling state)
)
CHECKPOINT_THRESHOLD = 3  # Checkpoint after 3 events


def benchmark_rolling_state():
    """Benchmark rolling state performance on fresh login with real GDrive."""
    os.environ["PRE_SYNC"] = "false"

    total_events = EVENTS_BEFORE_CHECKPOINT + EVENTS_AFTER_CHECKPOINT

    print("=" * 60)
    print("ROLLING STATE BENCHMARK (GDrive)")
    print("=" * 60)
    print("Configuration:")
    print(f"  DO Email: {EMAIL_DO}")
    print(f"  DS Email: {EMAIL_DS}")
    print(f"  Events before checkpoint: {EVENTS_BEFORE_CHECKPOINT}")
    print(f"  Events after checkpoint: {EVENTS_AFTER_CHECKPOINT}")
    print(f"  Total events: {total_events}")
    print(f"  Checkpoint threshold: {CHECKPOINT_THRESHOLD}")
    print(f"  Expected rolling state events: {EVENTS_AFTER_CHECKPOINT}")
    print()

    # Clean start - delete any existing syftboxes
    print("Cleaning up existing syftboxes...")
    ds, do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,
        use_in_memory_cache=False,
    )
    ds.delete_syftbox()
    do.delete_syftbox()

    # Fresh pair for event submission
    print("Creating fresh DS/DO pair...")
    ds, do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    # PHASE 1: Submit events to trigger checkpoint creation
    print(
        f"\n[PHASE 1] Submitting {EVENTS_BEFORE_CHECKPOINT} file changes to trigger checkpoint..."
    )
    submit_start = time.time()

    for i in range(EVENTS_BEFORE_CHECKPOINT):
        ds._send_file_change(
            f"{do.email}/file_{i:03d}.txt",
            f"content for file {i} - benchmark rolling state test",
        )
        print(f"  Submitted file {i + 1}/{EVENTS_BEFORE_CHECKPOINT}")

    # Wait for Google Drive to propagate
    print("\nWaiting for Google Drive propagation...")
    time.sleep(5)

    # DO syncs and creates checkpoint (this initializes rolling state)
    print("\nDO syncing (with auto-checkpoint)...")
    sync_start = time.time()
    do.sync(auto_checkpoint=True, checkpoint_threshold=CHECKPOINT_THRESHOLD)
    do_sync_time_phase1 = time.time() - sync_start
    print(f"DO sync complete: {do_sync_time_phase1:.2f}s")

    # Verify checkpoint was created and rolling state initialized
    checkpoint_after_phase1 = do._connection_router.get_latest_checkpoint()
    print("\nAfter Phase 1:")
    print(f"  Checkpoint exists: {checkpoint_after_phase1 is not None}")
    print(
        f"  Rolling state initialized: {do.datasite_owner_syncer._rolling_state is not None}"
    )

    # PHASE 2: Submit more events AFTER checkpoint (these go into rolling state)
    print(
        f"\n[PHASE 2] Submitting {EVENTS_AFTER_CHECKPOINT} file changes after checkpoint..."
    )

    for i in range(EVENTS_BEFORE_CHECKPOINT, total_events):
        ds._send_file_change(
            f"{do.email}/file_{i:03d}.txt",
            f"content for file {i} - benchmark rolling state test (after checkpoint)",
        )
        print(f"  Submitted file {i + 1}/{total_events}")

    # Wait for Google Drive to propagate
    print("\nWaiting for Google Drive propagation...")
    time.sleep(5)

    # DO syncs again - these events should go into rolling state
    print("\nDO syncing (events go into rolling state)...")
    sync_start = time.time()
    do.sync(
        auto_checkpoint=False
    )  # Don't auto-checkpoint, we want events in rolling state
    do_sync_time_phase2 = time.time() - sync_start
    print(f"DO sync complete: {do_sync_time_phase2:.2f}s")

    submit_time = time.time() - submit_start
    do_sync_time = do_sync_time_phase1 + do_sync_time_phase2

    # Verify checkpoint and rolling state were created
    checkpoint = do._connection_router.get_latest_checkpoint()
    rolling_state = do._connection_router.get_rolling_state()

    print(f"\nState after {total_events} events:")
    print(f"  Checkpoint exists: {checkpoint is not None}")
    if checkpoint:
        print(f"  Checkpoint files: {len(checkpoint.files)}")
        print(f"  Checkpoint timestamp: {checkpoint.timestamp}")
    print(f"  Rolling state exists: {rolling_state is not None}")
    if rolling_state:
        print(f"  Rolling state events: {rolling_state.event_count}")
        print(
            f"  Rolling state base checkpoint: {rolling_state.base_checkpoint_timestamp}"
        )
    else:
        print("  WARNING: Rolling state is None - events after checkpoint not tracked!")

    # Wait for Google Drive to propagate checkpoint and rolling state
    print("\nWaiting for Google Drive propagation of checkpoint/rolling state...")
    time.sleep(5)

    # Create a fresh DO manager to simulate initial sync from a new login
    print("\n" + "-" * 60)
    print("SIMULATING FRESH LOGIN")
    print("-" * 60)

    _, fresh_do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
        clear_caches=True,
    )

    # Verify we're using a different cache directory
    assert (
        fresh_do.datasite_owner_syncer.event_cache.file_connection.base_dir
        != do.datasite_owner_syncer.event_cache.file_connection.base_dir
    )

    # Benchmark fresh login sync
    print("\nStarting fresh login sync...")
    sync_start = time.time()
    fresh_do.sync(auto_checkpoint=False)
    fresh_sync_time = time.time() - sync_start

    # Get cache state
    cache = fresh_do.datasite_owner_syncer.event_cache
    files_in_cache = len(cache.file_hashes)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Events before checkpoint:       {EVENTS_BEFORE_CHECKPOINT}")
    print(f"Events after checkpoint:        {EVENTS_AFTER_CHECKPOINT}")
    print(f"Total events submitted:         {total_events}")
    print(f"Checkpoint threshold:           {CHECKPOINT_THRESHOLD}")
    print(
        f"Rolling state events:           {rolling_state.event_count if rolling_state else 0}"
    )
    print()
    print("Timing:")
    print(f"  Total submission + sync:      {submit_time:.2f}s")
    print(f"  DO sync (phase 1 + 2):        {do_sync_time:.2f}s")
    print(f"  Fresh login sync:             {fresh_sync_time:.2f}s")
    print()
    print(f"Files in cache after sync:      {files_in_cache}")
    print()

    return {
        "total_events": total_events,
        "events_before_checkpoint": EVENTS_BEFORE_CHECKPOINT,
        "events_after_checkpoint": EVENTS_AFTER_CHECKPOINT,
        "submit_time": submit_time,
        "do_sync_time": do_sync_time,
        "fresh_sync_time": fresh_sync_time,
        "files_in_cache": files_in_cache,
        "checkpoint_files": len(checkpoint.files) if checkpoint else 0,
        "rolling_state_events": rolling_state.event_count if rolling_state else 0,
    }


if __name__ == "__main__":
    results = benchmark_rolling_state()
    print("\nBenchmark complete!")
    print(f"Results: {results}")
