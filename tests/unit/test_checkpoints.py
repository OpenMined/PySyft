"""
Unit tests for checkpoint functionality.

Tests checkpoint creation, restore, threshold-based creation,
compacting (with merges, overwrites, deletions), deduplication,
and checkpoint methods.
"""

import time
from pathlib import Path
from uuid import uuid4

from syft_client.sync.checkpoints.checkpoint import (
    Checkpoint,
    IncrementalCheckpoint,
)
from syft_client.sync.checkpoints.rolling_state import RollingState
from syft_client.sync.events.file_change_event import FileChangeEvent
from syft_client.sync.syftbox_manager import SyftboxManager
from tests.unit.utils import get_mock_event


def test_checkpoint_create_and_restore():
    """Test that checkpoints can be created and restored."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Send some file changes to build state (path format: email/filename)
    ds_manager._send_file_change(f"{do_email}/test1.txt", "Content 1")
    ds_manager._send_file_change(f"{do_email}/test2.txt", "Content 2")
    do_manager.sync(auto_checkpoint=False)

    # Verify files are in cache
    do_cache = do_manager.datasite_owner_syncer.event_cache
    assert len(do_cache.file_hashes) == 2

    # Create checkpoint
    checkpoint = do_manager.create_checkpoint()

    assert checkpoint is not None
    assert len(checkpoint.files) == 2
    assert checkpoint.email == do_manager.email

    # Verify checkpoint is stored
    assert do_manager._connection_router.get_latest_checkpoint() is not None

    # Get latest checkpoint
    latest_checkpoint = do_manager._connection_router.get_latest_checkpoint()
    assert latest_checkpoint is not None
    assert latest_checkpoint.timestamp == checkpoint.timestamp


def test_checkpoint_should_create():
    """Test should_create_checkpoint logic."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # No events - should not create checkpoint with low threshold
    assert not do_manager.should_create_checkpoint(threshold=1)

    # Send file changes
    for i in range(5):
        ds_manager._send_file_change(f"{do_email}/test{i}.txt", f"Content {i}")
    do_manager.sync(auto_checkpoint=False)

    # Now we have 5 events - check thresholds
    assert do_manager.should_create_checkpoint(threshold=3)
    assert do_manager.should_create_checkpoint(threshold=5)
    assert not do_manager.should_create_checkpoint(threshold=10)


def test_checkpoint_try_create():
    """Test try_create_checkpoint only creates when threshold exceeded."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Send 3 file changes
    for i in range(3):
        ds_manager._send_file_change(f"{do_email}/test{i}.txt", f"Content {i}")
    do_manager.sync(auto_checkpoint=False)

    # With high threshold, should not create
    result = do_manager.try_create_checkpoint(threshold=10)
    assert result is None

    assert do_manager._connection_router.get_incremental_checkpoint_count() == 0

    # With low threshold, should create incremental checkpoint
    result = do_manager.try_create_checkpoint(threshold=2)
    assert result is not None
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 1


def test_checkpoint_restore_on_sync():
    """Test that sync uses checkpoint for initial state restore."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Send file changes and create initial state (path format: email/filename)
    ds_manager._send_file_change(f"{do_email}/test1.txt", "Content 1")
    ds_manager._send_file_change(f"{do_email}/test2.txt", "Content 2")
    do_manager.sync(auto_checkpoint=False)

    # Create checkpoint
    checkpoint = do_manager.create_checkpoint()
    assert len(checkpoint.files) == 2

    # Clear cache to simulate fresh login
    do_manager.datasite_owner_syncer.event_cache.clear_cache()
    do_manager.datasite_owner_syncer.initial_sync_done = False

    # Sync should restore from checkpoint
    do_manager.sync(auto_checkpoint=False)

    # Verify state was restored
    do_cache = do_manager.datasite_owner_syncer.event_cache
    assert len(do_cache.file_hashes) == 2


def test_checkpoint_events_since():
    """Test getting events since checkpoint timestamp."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Send initial files and sync (path format: email/filename)
    ds_manager._send_file_change(f"{do_email}/test1.txt", "Content 1")
    do_manager.sync(auto_checkpoint=False)

    # Create checkpoint
    checkpoint = do_manager.create_checkpoint()

    # Send more files after checkpoint
    ds_manager._send_file_change(f"{do_email}/test2.txt", "Content 2")
    ds_manager._send_file_change(f"{do_email}/test3.txt", "Content 3")
    do_manager.sync(auto_checkpoint=False)

    # Count events since checkpoint
    events_count = do_manager._connection_router.get_events_count_since_checkpoint(
        checkpoint.last_event_timestamp
    )
    # Should have 2 new events (test2.txt and test3.txt)
    assert events_count >= 2

    # Get events since checkpoint
    events_messages = do_manager._connection_router.get_events_messages_since_timestamp(
        checkpoint.last_event_timestamp
    )
    assert len(events_messages) >= 1


def test_checkpoint_excludes_datasets():
    """Test that checkpoints do not include files under syft_datasets folder."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Send regular file changes
    ds_manager._send_file_change(
        f"{do_email}/public/regular_file.txt", "Regular content"
    )
    ds_manager._send_file_change(
        f"{do_email}/public/another_file.txt", "Another content"
    )
    do_manager.sync(auto_checkpoint=False)

    # Manually write dataset files directly to the file_connection
    # to simulate dataset files in the syft_datasets folder
    do_cache = do_manager.datasite_owner_syncer.event_cache
    do_cache.file_connection.write_file(
        "public/syft_datasets/my_dataset/data.csv", "dataset,content\n1,2"
    )
    do_cache.file_connection.write_file(
        "public/syft_datasets/my_dataset/metadata.json", '{"name": "test"}'
    )

    # Verify regular files are in cache (from events)
    assert len(do_cache.file_hashes) == 2

    # Create checkpoint
    checkpoint = do_manager.create_checkpoint()

    # Verify checkpoint only contains regular files, NOT dataset files
    assert checkpoint is not None
    checkpoint_paths = [f.path for f in checkpoint.files]

    # Should have 2 regular files
    assert len(checkpoint.files) == 2

    # Regular files should be in checkpoint
    assert any("regular_file.txt" in p for p in checkpoint_paths)
    assert any("another_file.txt" in p for p in checkpoint_paths)

    # Dataset files should NOT be in checkpoint
    assert not any("syft_datasets" in p for p in checkpoint_paths)
    assert not any("data.csv" in p for p in checkpoint_paths)
    assert not any("metadata.json" in p for p in checkpoint_paths)


def test_compact_with_existing_full_checkpoint():
    """Test that compacting merges existing full checkpoint with incremental checkpoints."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Step 1: Create initial files and make a full checkpoint
    ds_manager._send_file_change(f"{do_email}/file1.txt", "content1")
    ds_manager._send_file_change(f"{do_email}/file2.txt", "content2")
    do_manager.sync(auto_checkpoint=False)

    # Create full checkpoint (contains file1 and file2)
    full_checkpoint = do_manager.create_checkpoint()
    assert len(full_checkpoint.files) == 2
    assert do_manager._connection_router.get_latest_checkpoint() is not None

    # Step 2: Create new files and make incremental checkpoints
    # Send 3 new files (file3, file4, file5)
    ds_manager._send_file_change(f"{do_email}/file3.txt", "content3")
    ds_manager._send_file_change(f"{do_email}/file4.txt", "content4")
    ds_manager._send_file_change(f"{do_email}/file5.txt", "content5")
    do_manager.sync(auto_checkpoint=False)

    # Create first incremental checkpoint (should have 3 files)
    do_manager.try_create_checkpoint(threshold=3)
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 1

    # Send 3 more files (file6, file7, file8)
    ds_manager._send_file_change(f"{do_email}/file6.txt", "content6")
    ds_manager._send_file_change(f"{do_email}/file7.txt", "content7")
    ds_manager._send_file_change(f"{do_email}/file8.txt", "content8")
    do_manager.sync(auto_checkpoint=False)

    # Create second incremental checkpoint
    do_manager.try_create_checkpoint(threshold=3)
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 2

    # Step 3: Manually call compact_checkpoints
    # This should merge:
    # - Full checkpoint (file1, file2)
    # - Incremental checkpoint #1 (file3, file4, file5)
    # - Incremental checkpoint #2 (file6, file7, file8)
    # = New full checkpoint with all 8 files
    compacted = do_manager.datasite_owner_syncer.compact_checkpoints()

    # Verify compacted checkpoint has all 8 files
    assert len(compacted.files) == 8
    checkpoint_paths = {f.path for f in compacted.files}

    # All files should be present (paths don't include email prefix in checkpoint)
    assert "file1.txt" in checkpoint_paths
    assert "file2.txt" in checkpoint_paths
    assert "file3.txt" in checkpoint_paths
    assert "file4.txt" in checkpoint_paths
    assert "file5.txt" in checkpoint_paths
    assert "file6.txt" in checkpoint_paths
    assert "file7.txt" in checkpoint_paths
    assert "file8.txt" in checkpoint_paths

    # Verify incremental checkpoints are deleted
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 0

    # Verify full checkpoint is updated
    assert do_manager._connection_router.get_latest_checkpoint() is not None


def test_compact_with_no_existing_full_checkpoint():
    """Test that compacting works when there's no existing full checkpoint."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Create incremental checkpoints WITHOUT a full checkpoint first
    # First incremental checkpoint
    ds_manager._send_file_change(f"{do_email}/file1.txt", "content1")
    ds_manager._send_file_change(f"{do_email}/file2.txt", "content2")
    do_manager.sync(auto_checkpoint=False)
    do_manager.try_create_checkpoint(threshold=2)
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 1

    # Second incremental checkpoint
    ds_manager._send_file_change(f"{do_email}/file3.txt", "content3")
    ds_manager._send_file_change(f"{do_email}/file4.txt", "content4")
    do_manager.sync(auto_checkpoint=False)
    do_manager.try_create_checkpoint(threshold=2)
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 2

    # Compact (without existing full checkpoint)
    compacted = do_manager.datasite_owner_syncer.compact_checkpoints()

    # Verify compacted checkpoint has all 4 files
    assert len(compacted.files) == 4
    checkpoint_paths = {f.path for f in compacted.files}
    assert "file1.txt" in checkpoint_paths
    assert "file2.txt" in checkpoint_paths
    assert "file3.txt" in checkpoint_paths
    assert "file4.txt" in checkpoint_paths

    # Verify incremental checkpoints are deleted
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 0


def test_incremental_checkpoint_deduplication():
    """Test that incremental checkpoint deduplicates events by path."""
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )

    do_email = do_manager.email

    # Manually build rolling state with duplicate paths (bypasses hash conflict check)
    rs = RollingState(
        email=do_email,
        base_checkpoint_timestamp=0.0,
    )

    event1 = get_mock_event(f"{do_email}/changing.txt")
    event1.content = "version1"
    rs.add_event(event1)

    event2 = get_mock_event(f"{do_email}/changing.txt")
    event2.content = "version2"
    rs.add_event(event2)

    event3 = get_mock_event(f"{do_email}/changing.txt")
    event3.content = "version3"
    rs.add_event(event3)

    stable_event = get_mock_event(f"{do_email}/stable.txt")
    stable_event.content = "stable_content"
    rs.add_event(stable_event)

    # Rolling state should have 2 events (deduplicated at add time)
    assert rs.event_count == 2

    # Inject rolling state into the syncer and create incremental checkpoint
    do_manager.datasite_owner_syncer._rolling_state = rs
    inc_cp = do_manager.datasite_owner_syncer.create_incremental_checkpoint()

    # Should only have 2 events (1 for changing.txt latest, 1 for stable.txt)
    assert inc_cp.event_count == 2

    # The changing.txt event should have the latest content
    changing_events = [
        e for e in inc_cp.events if "changing.txt" in str(e.path_in_datasite)
    ]
    assert len(changing_events) == 1
    assert changing_events[0].content == "version3"


def test_compact_with_file_overwrites_across_incrementals():
    """Test compacting where the same file is modified across incremental checkpoints."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Create full checkpoint with file1
    ds_manager._send_file_change(f"{do_email}/file1.txt", "original")
    do_manager.sync(auto_checkpoint=False)
    do_manager.create_checkpoint()

    # Manually construct incremental #1: modify file1 + add file2
    inc1_events = [
        FileChangeEvent(
            id=uuid4(),
            path_in_datasite=Path("file1.txt"),
            datasite_email=do_email,
            content="modified_v2",
            old_hash=None,
            new_hash="hash_v2",
            is_deleted=False,
            submitted_timestamp=time.time(),
            timestamp=time.time(),
        ),
        FileChangeEvent(
            id=uuid4(),
            path_in_datasite=Path("file2.txt"),
            datasite_email=do_email,
            content="content2",
            old_hash=None,
            new_hash="hash_file2",
            is_deleted=False,
            submitted_timestamp=time.time(),
            timestamp=time.time(),
        ),
    ]
    inc_cp1 = IncrementalCheckpoint(
        email=do_email,
        sequence_number=1,
        events=inc1_events,
    )
    do_manager._connection_router.upload_incremental_checkpoint(inc_cp1)

    # Manually construct incremental #2: modify file1 again
    inc2_events = [
        FileChangeEvent(
            id=uuid4(),
            path_in_datasite=Path("file1.txt"),
            datasite_email=do_email,
            content="modified_v3",
            old_hash="hash_v2",
            new_hash="hash_v3",
            is_deleted=False,
            submitted_timestamp=time.time(),
            timestamp=time.time(),
        ),
    ]
    inc_cp2 = IncrementalCheckpoint(
        email=do_email,
        sequence_number=2,
        events=inc2_events,
    )
    do_manager._connection_router.upload_incremental_checkpoint(inc_cp2)

    assert do_manager._connection_router.get_incremental_checkpoint_count() == 2

    # Compact
    compacted = do_manager.datasite_owner_syncer.compact_checkpoints()

    # Should have 2 files: file1 (latest version) and file2
    assert len(compacted.files) == 2
    file1 = next(f for f in compacted.files if f.path == "file1.txt")
    assert file1.content == "modified_v3"


def test_compact_with_file_deletions():
    """Test compacting excludes files marked as deleted in incremental checkpoints."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email

    # Create full checkpoint with file1 and file2
    ds_manager._send_file_change(f"{do_email}/file1.txt", "content1")
    ds_manager._send_file_change(f"{do_email}/file2.txt", "content2")
    do_manager.sync(auto_checkpoint=False)
    do_manager.create_checkpoint()

    # Incremental #1: add file3
    ds_manager._send_file_change(f"{do_email}/file3.txt", "content3")
    do_manager.sync(auto_checkpoint=False)
    do_manager.try_create_checkpoint(threshold=1)

    # Manually create incremental #2 with a deletion event for file1
    deletion_event = FileChangeEvent(
        id=uuid4(),
        path_in_datasite=Path("file1.txt"),
        datasite_email=do_email,
        content=None,
        old_hash="some_hash",
        new_hash=None,
        is_deleted=True,
        submitted_timestamp=time.time(),
        timestamp=time.time(),
    )
    inc_cp = IncrementalCheckpoint(
        email=do_email,
        sequence_number=do_manager._connection_router.get_next_incremental_sequence_number(),
        events=[deletion_event],
    )
    do_manager._connection_router.upload_incremental_checkpoint(inc_cp)

    # Compact
    compacted = do_manager.datasite_owner_syncer.compact_checkpoints()

    # Should have 2 files: file2 and file3 (file1 was deleted)
    checkpoint_paths = {f.path for f in compacted.files}
    assert "file1.txt" not in checkpoint_paths
    assert "file2.txt" in checkpoint_paths
    assert "file3.txt" in checkpoint_paths
    assert len(compacted.files) == 2


def test_try_create_checkpoint_triggers_compacting():
    """Test that try_create_checkpoint triggers compacting when both thresholds are met."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    do_email = do_manager.email
    syncer = do_manager.datasite_owner_syncer

    # Create 2 incremental checkpoints (use high compacting_threshold to prevent early compacting)
    for batch in range(2):
        for i in range(3):
            ds_manager._send_file_change(
                f"{do_email}/batch{batch}_file{i}.txt", f"content_{batch}_{i}"
            )
        do_manager.sync(auto_checkpoint=False)
        syncer.try_create_checkpoint(threshold=3, compacting_threshold=999)

    assert do_manager._connection_router.get_incremental_checkpoint_count() == 2
    assert do_manager._connection_router.get_latest_checkpoint() is None

    # Now send more events and call try_create_checkpoint with low compacting threshold
    for i in range(3):
        ds_manager._send_file_change(f"{do_email}/batch2_file{i}.txt", f"content_2_{i}")
    do_manager.sync(auto_checkpoint=False)

    # This should create incremental #3 AND trigger compacting (threshold=3, compacting=3)
    result = syncer.try_create_checkpoint(threshold=3, compacting_threshold=3)

    # Result should be a full Checkpoint (from compacting), not IncrementalCheckpoint
    assert isinstance(result, Checkpoint)

    # All incrementals should be deleted, full checkpoint created
    assert do_manager._connection_router.get_incremental_checkpoint_count() == 0
    assert do_manager._connection_router.get_latest_checkpoint() is not None
    assert len(result.files) == 9  # 3 batches × 3 files


def test_local_do_changes_end_up_in_incremental_checkpoint():
    """Regression test: local DO file changes must flow into rolling state
    and then into incremental checkpoints.

    Without the fix in process_local_changes, local events only went to the
    GDrive event log but never reached rolling state → never in checkpoints.
    A new client doing initial sync would miss them permanently.
    """
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    # Make local DO changes (direct filesystem writes, not via _send_file_change)
    for i in range(3):
        f = do_manager.syftbox_folder / do_manager.email / f"local_{i}.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"local content {i}")

    # Force checkpoint with threshold=1
    do_manager.sync(auto_checkpoint=True, checkpoint_threshold=1)

    # An incremental checkpoint should have been created from local changes
    incremental_cps = do_manager._connection_router.get_all_incremental_checkpoints()
    assert len(incremental_cps) >= 1, (
        "No incremental checkpoint created. Local DO events never reached "
        "rolling state, so should_create_checkpoint never fired."
    )

    # Local file paths must appear in at least one incremental checkpoint
    paths_in_checkpoints = set()
    for cp in incremental_cps:
        for event in cp.events:
            paths_in_checkpoints.add(str(event.path_in_datasite))

    for i in range(3):
        assert f"local_{i}.txt" in paths_in_checkpoints, (
            f"local_{i}.txt missing from incremental checkpoints. "
            f"process_local_changes did not add it to rolling state."
        )
