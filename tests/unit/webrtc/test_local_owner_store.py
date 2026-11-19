"""LocalOwnerStoreConnection, and a DO that restores its state from it."""

from syft.sync.connections.webrtc.local_owner_store import (
    CHECKPOINTS_FOLDER,
    OWNER_STORE_PATH,
    LocalOwnerStoreConnection,
)
from syft.sync.events.file_change_event import FileChangeEventsMessageFileName
from syft.sync.syftbox_manager import SyftboxManager


def event_name(timestamp: float) -> str:
    return FileChangeEventsMessageFileName(timestamp=timestamp).as_string()


def test_event_handles_newer_than_timestamp_oldest_first(tmp_path):
    store = LocalOwnerStoreConnection(tmp_path)
    store.append_event(event_name(3.0), b"third")
    store.append_event(event_name(1.0), b"first")
    store.append_event(event_name(2.0), b"second")

    handles = store.list_event_handles(since_timestamp=1.0)
    assert [store.fetch(handle) for handle in handles] == [b"second", b"third"]


def test_new_checkpoint_replaces_older(tmp_path):
    store = LocalOwnerStoreConnection(tmp_path)
    store.put_checkpoint("checkpoint_1.0.tar.gz", b"old")
    store.put_checkpoint("checkpoint_2.0.tar.gz", b"new")

    assert store.latest_checkpoint() == b"new"
    assert len(list((tmp_path / CHECKPOINTS_FOLDER).iterdir())) == 1


def test_incrementals_follow_sequence_order(tmp_path):
    store = LocalOwnerStoreConnection(tmp_path)
    # As text, "10" sorts before "9".
    store.put_incremental("incremental_checkpoint_10_2.0.tar.gz", b"tenth")
    store.put_incremental("incremental_checkpoint_9_1.0.tar.gz", b"ninth")

    assert store.all_incrementals() == [b"ninth", b"tenth"]
    assert store.next_incremental_sequence() == 11
    store.delete_incrementals()
    assert store.incremental_count() == 0


def test_delete_all_removes_store(tmp_path):
    folder = tmp_path / "store"
    store = LocalOwnerStoreConnection(folder)
    store.append_event(event_name(1.0), b"event")
    store.put_rolling_state("rolling_state_1.0.tar.gz", b"state")

    store.delete_all()

    assert not folder.exists()


def test_do_restores_owner_state_from_local_store():
    do_config, ds_config = SyftboxManager._mock_pair_configs(use_in_memory_cache=True)
    do_connections, ds_connections = SyftboxManager._mock_drive_connections(
        do_config, ds_config
    )
    local_store = LocalOwnerStoreConnection(do_config.syftbox_folder / OWNER_STORE_PATH)
    ds_manager, do_manager = SyftboxManager._test_pair(
        do_config,
        do_connections.model_copy(update={"owner_store_con": local_store}),
        ds_config,
        ds_connections,
        add_peers=True,
    )
    do_syncer = do_manager.datasite_owner_syncer
    do_syncer.perm_context.open(".").grant_write_access(ds_manager.email)
    ds_manager._send_file_change(f"{do_manager.email}/test1.txt", "Content 1")
    ds_manager._send_file_change(f"{do_manager.email}/test2.txt", "Content 2")
    do_manager.sync(auto_checkpoint=False)
    do_manager.create_checkpoint()

    # A new login: the cache starts empty, and the first sync restores it.
    do_syncer.event_cache.clear_cache()
    do_syncer.initial_sync_done = False
    do_manager.sync(auto_checkpoint=False)

    assert len(do_syncer.event_cache.file_hashes) == 2
