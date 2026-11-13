"""With ``freeze_private_datasets`` on, a file in an immutable collection
keeps the first copy that the DS received from a DO."""

import time
import uuid
from pathlib import Path

from syft.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
)
from syft.sync.sync.caches.datasite_watcher_cache import (
    DataSiteWatcherCache,
    DataSiteWatcherCacheConfig,
)
from syft.sync.sync.collection_spec import CollectionSyncSpec
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.utils.syftbox_utils import get_event_hash_from_content
from tests.unit.utils import router_with_mock_connection

DS = "ds@example.com"
DO = "do@example.com"
PRIVATE_SPEC = CollectionSyncSpec.private(
    "syft_privatecollection", Path("private/syft_datasets")
)
PRIVATE_FILE = "private/syft_datasets/v1/my_ds/data.csv"


def _event(
    path: str, content: str | None = None, datasite_email: str = DO
) -> FileChangeEvent:
    """A change of ``path`` to ``content``, or a delete if ``content`` is None."""
    return FileChangeEvent(
        id=uuid.uuid4(),
        datasite_email=datasite_email,
        path_in_datasite=Path(path),
        content=content,
        new_hash=None if content is None else get_event_hash_from_content(content),
        is_deleted=content is None,
        submitted_timestamp=time.time(),
        timestamp=time.time(),
    )


def _cache(freeze: bool) -> DataSiteWatcherCache:
    return DataSiteWatcherCache.from_config(
        DataSiteWatcherCacheConfig(
            email=DS,
            collection_specs=[PRIVATE_SPEC],
            freeze_private_datasets=freeze,
        ),
        router_with_mock_connection(DS),
    )


def _apply(cache: DataSiteWatcherCache, *events: FileChangeEvent) -> None:
    cache.apply_event_message(FileChangeEventsMessage(events=list(events)), DO)


def test_frozen_file_keeps_first_copy(caplog):
    cache = _cache(freeze=True)
    _apply(cache, _event(PRIVATE_FILE, "first"))

    _apply(cache, _event(PRIVATE_FILE, "second"), _event(PRIVATE_FILE))

    assert cache.file_connection.read_file(f"{DO}/{PRIVATE_FILE}") == "first"
    assert "immutable collection" in caplog.text


def test_file_outside_immutable_collections_changes():
    cache = _cache(freeze=True)
    path = "public/syft_datasets/my_ds/data.csv"
    _apply(cache, _event(path, "first"))

    _apply(cache, _event(path, "second"))

    assert cache.file_connection.read_file(f"{DO}/{path}") == "second"


def test_unfrozen_cache_changes_immutable_collection():
    cache = _cache(freeze=False)
    _apply(cache, _event(PRIVATE_FILE, "first"))

    _apply(cache, _event(PRIVATE_FILE, "second"))

    assert cache.file_connection.read_file(f"{DO}/{PRIVATE_FILE}") == "second"


def test_refused_change_not_restored_after_restart():
    """Over the mock Drive: the saved state of the DS holds only the first copy."""
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    syncer = ds.datasite_watcher_syncer
    assert syncer is not None
    syncer.datasite_watcher_cache.freeze_private_datasets = True
    syncer.datasite_watcher_cache.collection_specs = [PRIVATE_SPEC]
    first = _event(PRIVATE_FILE, "first", datasite_email=do.email)
    for event in (first, _event(PRIVATE_FILE, "second", datasite_email=do.email)):
        do._connection_router.owner_write_event_messages_to_outbox(
            ds.email, FileChangeEventsMessage(events=[event])
        )
        ds.sync()

    restarted_syncer = ds._copy().datasite_watcher_syncer
    assert restarted_syncer is not None

    restarted = restarted_syncer.datasite_watcher_cache
    assert restarted.file_hashes[Path(f"{do.email}/{PRIVATE_FILE}")] == first.new_hash
