import json
import time
import uuid

import pytest

from syft.sync.connections.types import MessageKind
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.events.file_change_event import FileChangeEvent, FileChangeEventsMessage
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft.sync.sync.caches import datasite_watcher_cache
from syft.sync.sync.caches.datasite_watcher_cache import DataSiteWatcherCache
from syft.sync.utils.syftbox_utils import compress_data
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
    monkeypatch.setattr(datasite_watcher_cache, "DEFAULT_CHECKPOINT_EVENT_THRESHOLD", 2)
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


def _encrypted_pair() -> tuple[SyftboxManager, SyftboxManager]:
    """A DS and a DO on disk that require encryption."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False, encryption=True
    )
    return ds_manager, do_manager


def _rebooted_config(
    ds_manager: SyftboxManager, tmp_path, persist_watcher_state: bool = True
) -> SyftboxManagerConfig:
    """The config of ``ds_manager`` on an empty folder, so it gets a new key."""
    return SyftboxManagerConfig._base_config_for_testing(
        email=ds_manager.email,
        syftbox_folder=tmp_path / "SyftBox",
        has_ds_role=True,
        use_in_memory_cache=False,
        persist_watcher_state=persist_watcher_state,
        encryption_policy=EncryptionPolicy.REQUIRED,
    )


def test_restart_with_new_key_names_key_file(tmp_path):
    ds_manager, do_manager = _encrypted_pair()
    _send_to_ds(do_manager, ds_manager, [get_mock_event(f"{do_manager.email}/a.txt")])
    ds_manager.sync()

    with pytest.raises(ValueError, match="key_file"):
        SyftboxManager.from_config(
            _rebooted_config(ds_manager, tmp_path),
            ds_manager._connection_router.connections,
        )


def test_watcher_state_off_neither_saved_nor_restored(tmp_path):
    """An enclave keeps no watcher state: it saves none, and a boot with a new
    key reads none, even where an earlier run saved some."""
    ds_manager, do_manager = _encrypted_pair()
    router = ds_manager._connection_router
    _send_to_ds(do_manager, ds_manager, [get_mock_event(f"{do_manager.email}/a.txt")])
    ds_manager.sync()
    saved = router.watcher_load_rolling_state(do_manager.email.casefold())
    assert saved is not None

    _watcher_cache(ds_manager).persist_watcher_state = False
    _send_to_ds(do_manager, ds_manager, [get_mock_event(f"{do_manager.email}/b.txt")])
    ds_manager.sync()
    assert router.watcher_load_rolling_state(do_manager.email.casefold()) == saved

    SyftboxManager.from_config(
        _rebooted_config(ds_manager, tmp_path, persist_watcher_state=False),
        router.connections,
    )


def test_in_memory_cache_restores_watcher_state():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    event = get_mock_event(f"{do_manager.email}/a.txt")
    _send_to_ds(do_manager, ds_manager, [event])
    ds_manager.sync()
    config = SyftboxManagerConfig._base_config_for_testing(
        email=ds_manager.email, has_ds_role=True
    )

    restarted = SyftboxManager.from_config(
        config, ds_manager._connection_router.connections
    )

    restored = _watcher_cache(restarted)
    assert restored.file_hashes[event.path_in_syftbox] == event.new_hash
    assert (
        restored.last_msg_id_per_peer == _watcher_cache(ds_manager).last_msg_id_per_peer
    )


def test_binary_file_restored_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(datasite_watcher_cache, "DEFAULT_CHECKPOINT_EVENT_THRESHOLD", 1)
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
    monkeypatch.setattr(datasite_watcher_cache, "DEFAULT_CHECKPOINT_EVENT_THRESHOLD", 1)
    ds_manager, do_manager = _pair_on_disk()
    event = get_mock_event(f"{do_manager.email.upper()}/result.txt")
    _send_to_ds(do_manager, ds_manager, [event])
    ds_manager.sync()

    restarted = _restart_on_empty_folder(ds_manager, tmp_path)

    assert (restarted.syftbox_folder / event.path_in_syftbox).exists()


def test_do_state_saved_under_email_without_case(monkeypatch):
    monkeypatch.setattr(datasite_watcher_cache, "DEFAULT_CHECKPOINT_EVENT_THRESHOLD", 1)
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        email1="Owner@Example.com", use_in_memory_cache=False
    )
    _send_to_ds(do_manager, ds_manager, [get_mock_event("Owner@Example.com/a.txt")])

    ds_manager.sync()

    router = ds_manager._connection_router
    assert router.watcher_peers_with_saved_state() == ["owner@example.com"]


def test_message_with_change_without_content_skipped():
    ds_manager, do_manager = _pair_on_disk()
    bad_message = FileChangeEventsMessage(
        events=[
            get_mock_event(f"{do_manager.email}/no_content.txt"),
            get_mock_event(f"{do_manager.email}/beside_bad_change.txt"),
        ]
    )
    # The model refuses a change without content, so only raw bytes carry one.
    data = json.loads(bad_message.model_dump_json())
    data["events"][0]["content"] = None
    do_manager._connection_router.connections.p2p_con.send(
        ds_manager.email,
        bad_message.message_filepath.as_string(),
        compress_data(json.dumps(data).encode("utf-8")),
        MessageKind.FILE_EVENTS,
    )
    _send_to_ds(
        do_manager, ds_manager, [get_mock_event(f"{do_manager.email}/good.txt")]
    )

    with pytest.warns(UserWarning, match="Skipped unreadable outbox message"):
        ds_manager.sync()

    datasite = ds_manager.syftbox_folder / do_manager.email
    assert (datasite / "good.txt").exists()
    assert not (datasite / "beside_bad_change.txt").exists()
    assert not (datasite / "no_content.txt").exists()
