import time
import uuid

from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.events.file_change_event import FileChangeEvent, FileChangeEventsMessage
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft.sync.sync.caches import datasite_watcher_cache
from syft.sync.sync.caches.datasite_watcher_cache import DataSiteWatcherCache
from tests.unit.utils import get_mock_event


def _pair_on_disk() -> tuple[SyftboxManager, SyftboxManager]:
    return SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )


def _send_to_ds(
    do_manager: SyftboxManager,
    ds_manager: SyftboxManager,
    events: list[FileChangeEvent],
) -> None:
    do_manager._connection_router.owner_write_event_messages_to_outbox(
        ds_manager.email, FileChangeEventsMessage(events=events)
    )


def _deletion_of(event: FileChangeEvent) -> FileChangeEvent:
    return FileChangeEvent(
        id=uuid.uuid4(),
        path_in_datasite=event.path_in_datasite,
        datasite_email=event.datasite_email,
        old_hash=event.new_hash,
        is_deleted=True,
        submitted_timestamp=time.time(),
        timestamp=time.time(),
    )


def _watcher_cache(manager: SyftboxManager) -> DataSiteWatcherCache:
    syncer = manager.datasite_watcher_syncer
    assert syncer is not None
    return syncer.datasite_watcher_cache


def _restart_on_empty_folder(ds_manager: SyftboxManager, tmp_path) -> SyftboxManager:
    """A new DS manager on the Drive of ``ds_manager`` and an empty folder, as
    after a Colab runtime reset. It loads its peers, as a login does."""
    config = SyftboxManagerConfig._base_config_for_testing(
        email=ds_manager.email,
        syftbox_folder=tmp_path / "SyftBox",
        has_ds_role=True,
        use_in_memory_cache=False,
    )
    restarted = SyftboxManager.from_config(
        config, ds_manager._connection_router.connections
    )
    restarted.load_peers()
    return restarted


def test_restart_on_empty_folder_restores_files_and_cursor(tmp_path):
    ds_manager, do_manager = _pair_on_disk()
    event = get_mock_event(f"{do_manager.email}/result.txt")
    _send_to_ds(do_manager, ds_manager, [event])
    ds_manager.sync()

    restarted = _restart_on_empty_folder(ds_manager, tmp_path)

    restored = restarted.syftbox_folder / event.path_in_syftbox
    assert restored.read_text() == event.content
    cache = _watcher_cache(restarted)
    assert cache.sync_down(do_manager.email) == 0


def test_restart_after_checkpoint_keeps_deleted_file_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(datasite_watcher_cache, "WATCHER_CHECKPOINT_THRESHOLD", 2)
    ds_manager, do_manager = _pair_on_disk()
    kept = get_mock_event(f"{do_manager.email}/kept.txt")
    deleted = get_mock_event(f"{do_manager.email}/deleted.txt")
    _send_to_ds(do_manager, ds_manager, [kept, deleted])
    ds_manager.sync()
    router = ds_manager._connection_router
    assert router.watcher_load_checkpoint(do_manager.email) is not None
    _send_to_ds(do_manager, ds_manager, [_deletion_of(deleted)])
    ds_manager.sync()

    restarted = _restart_on_empty_folder(ds_manager, tmp_path)

    assert (restarted.syftbox_folder / kept.path_in_syftbox).exists()
    assert not (restarted.syftbox_folder / deleted.path_in_syftbox).exists()


def test_file_refused_by_write_filter_not_restored(tmp_path):
    ds_manager, do_manager = _pair_on_disk()
    refused = get_mock_event(f"{do_manager.email}/refused.txt")
    cache = _watcher_cache(ds_manager)
    cache.pre_write_filter = lambda path, is_delete: not path.endswith("refused.txt")
    _send_to_ds(do_manager, ds_manager, [refused])
    ds_manager.sync()

    restarted = _restart_on_empty_folder(ds_manager, tmp_path)

    assert not (restarted.syftbox_folder / refused.path_in_syftbox).exists()


def test_restart_with_new_key_restores_files_and_cursor(tmp_path):
    """An enclave gets a new key at each boot. Its watcher state stays
    readable, because it is not encrypted."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False, encryption=True
    )
    event = get_mock_event(f"{do_manager.email}/result.txt")
    _send_to_ds(do_manager, ds_manager, [event])
    ds_manager.sync()
    config = SyftboxManagerConfig._base_config_for_testing(
        email=ds_manager.email,
        syftbox_folder=tmp_path / "SyftBox",
        has_ds_role=True,
        use_in_memory_cache=False,
        encryption_policy=EncryptionPolicy.REQUIRED,
    )

    rebooted = SyftboxManager.from_config(
        config, ds_manager._connection_router.connections
    )

    restored = rebooted.syftbox_folder / event.path_in_syftbox
    assert restored.read_text() == event.content
    cursor = _watcher_cache(ds_manager).last_msg_id_per_peer
    assert _watcher_cache(rebooted).last_msg_id_per_peer == cursor


def test_binary_file_restored_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(datasite_watcher_cache, "WATCHER_CHECKPOINT_THRESHOLD", 1)
    ds_manager, do_manager = _pair_on_disk()
    event = get_mock_event(f"{do_manager.email}/image.bin").model_copy(
        update={"content": b"\x00\xff"}
    )
    _send_to_ds(do_manager, ds_manager, [event])
    ds_manager.sync()
    router = ds_manager._connection_router
    assert router.watcher_load_checkpoint(do_manager.email) is not None

    restarted = _restart_on_empty_folder(ds_manager, tmp_path)

    restored = restarted.syftbox_folder / event.path_in_syftbox
    assert restored.read_bytes() == b"\x00\xff"


def test_restore_writes_files_where_events_write(tmp_path, monkeypatch):
    """The DO's events may spell its email in another case than the DS's peer
    entry does. A restored file goes where the live event put it."""
    monkeypatch.setattr(datasite_watcher_cache, "WATCHER_CHECKPOINT_THRESHOLD", 1)
    ds_manager, do_manager = _pair_on_disk()
    event = get_mock_event(f"{do_manager.email.upper()}/result.txt")
    _send_to_ds(do_manager, ds_manager, [event])
    ds_manager.sync()

    restarted = _restart_on_empty_folder(ds_manager, tmp_path)

    assert (restarted.syftbox_folder / event.path_in_syftbox).exists()


def test_do_state_saved_under_email_without_case(monkeypatch):
    monkeypatch.setattr(datasite_watcher_cache, "WATCHER_CHECKPOINT_THRESHOLD", 1)
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        email1="Owner@Example.com", use_in_memory_cache=False
    )
    _send_to_ds(do_manager, ds_manager, [get_mock_event("Owner@Example.com/a.txt")])

    ds_manager.sync()

    router = ds_manager._connection_router
    assert router.watcher_peers_with_saved_state() == ["owner@example.com"]
