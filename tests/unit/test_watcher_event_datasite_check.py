"""Events pulled from a peer's outbox may only touch that peer's own datasite.

An event names its own target, and both ``datasite_email`` and
``path_in_datasite`` come from the peer. The watcher mirrors a peer at
``<syftbox>/<peer_email>/``; anything aimed elsewhere is dropped.
"""

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
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.utils.syftbox_utils import get_event_hash_from_content

VICTIM = "victim@example.com"
PEER = "peer@example.com"


def _event(
    datasite_email: str, path: str, content: str | None = "x", deleted: bool = False
) -> FileChangeEvent:
    return FileChangeEvent(
        id=uuid.uuid4(),
        datasite_email=datasite_email,
        path_in_datasite=Path(path),
        content=None if deleted else content,
        new_hash=None if deleted else get_event_hash_from_content(content),
        is_deleted=deleted,
        submitted_timestamp=time.time(),
        timestamp=time.time(),
    )


def _cache() -> DataSiteWatcherCache:
    return DataSiteWatcherCache.from_config(
        DataSiteWatcherCacheConfig(email=VICTIM, use_in_memory_cache=True)
    )


def _has(cache: DataSiteWatcherCache, path: str) -> bool:
    try:
        cache.file_connection.read_file(path)
        return True
    except Exception:
        return False


def test_event_for_peer_own_datasite_is_applied():
    cache = _cache()
    cache.apply_event_message(
        FileChangeEventsMessage(events=[_event(PEER, "data/file.txt")]), PEER
    )
    assert _has(cache, f"{PEER}/data/file.txt")


def test_event_aimed_at_victim_datasite_is_dropped(caplog):
    cache = _cache()
    cache.apply_event_message(
        FileChangeEventsMessage(
            events=[
                _event(
                    VICTIM,
                    "syft.pub.yaml",
                    "rules: [{pattern: '**', access: {read: ['*']}}]",
                )
            ]
        ),
        PEER,
    )
    assert not _has(cache, f"{VICTIM}/syft.pub.yaml")
    assert "may only change files under its own datasite" in caplog.text


def test_event_reaching_another_datasite_through_dotdot_is_dropped():
    cache = _cache()
    cache.apply_event_message(
        FileChangeEventsMessage(events=[_event(PEER, f"../{VICTIM}/syft.pub.yaml")]),
        PEER,
    )
    assert not _has(cache, f"{VICTIM}/syft.pub.yaml")
    assert not _has(cache, f"{PEER}/../{VICTIM}/syft.pub.yaml")


def test_deletion_aimed_at_victim_datasite_is_dropped():
    cache = _cache()
    cache.file_connection.write_file(f"{VICTIM}/keep.txt", "mine")
    cache.apply_event_message(
        FileChangeEventsMessage(events=[_event(VICTIM, "keep.txt", deleted=True)]),
        PEER,
    )
    assert _has(cache, f"{VICTIM}/keep.txt")


def test_email_comparison_ignores_case():
    cache = _cache()
    cache.apply_event_message(
        FileChangeEventsMessage(events=[_event(PEER.upper(), "file.txt")]), PEER
    )
    assert _has(cache, f"{PEER.upper()}/file.txt")


def test_forged_outbox_message_cannot_rewrite_ds_root_permissions():
    """Over the mock Drive: a DO publishes an event aimed at the DS's own datasite."""
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    own_rules = ds.syftbox_folder / ds.email / "syft.pub.yaml"
    own_rules.parent.mkdir(parents=True, exist_ok=True)
    own_rules.write_text("rules: []\n")

    forged = _event(
        ds.email, "syft.pub.yaml", "rules: [{pattern: '**', access: {read: ['*']}}]"
    )
    legitimate = _event(do.email, "public/readme.txt", "hello")
    do.datasite_owner_syncer.connection_router.owner_write_event_messages_to_outbox(
        ds.email, FileChangeEventsMessage(events=[forged, legitimate])
    )

    ds.sync()

    assert own_rules.read_text() == "rules: []\n"
    assert (ds.syftbox_folder / do.email / "public/readme.txt").read_text() == "hello"
