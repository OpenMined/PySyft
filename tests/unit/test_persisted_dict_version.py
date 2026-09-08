"""A persisted cache carries a version, and an unknown one resets the cache.

The client can rebuild every one of these caches from the events and the files,
so an unreadable cache costs a re-scan and nothing else. An unknown version
therefore starts empty instead of stopping the client.
"""

import json

from syft.sync.sync.caches.persisted_dict import (
    PERSISTED_DICT_VERSION,
    PersistedDict,
)


def _path(tmp_path):
    return tmp_path / "cache.json"


def test_a_saved_file_carries_the_version(tmp_path):
    d = PersistedDict(path=_path(tmp_path))
    d["a"] = "1"
    data = json.loads(_path(tmp_path).read_text())
    assert data["version"] == PERSISTED_DICT_VERSION
    assert data["entries"] == {"a": "1"}


def test_a_saved_file_loads_back(tmp_path):
    d = PersistedDict(path=_path(tmp_path))
    d["a"] = "1"
    assert PersistedDict(path=_path(tmp_path)).get("a") == "1"


def test_a_file_without_a_version_still_loads(tmp_path):
    # Written before the version field existed: a bare map of entries. Reading it
    # saves the user a full re-scan on the first run after an upgrade.
    _path(tmp_path).write_text(json.dumps({"a": "1", "b": "2"}))
    d = PersistedDict(path=_path(tmp_path))
    assert d.get("a") == "1"
    assert d.get("b") == "2"


def test_a_file_from_a_newer_client_starts_empty(tmp_path):
    _path(tmp_path).write_text(
        json.dumps({"version": PERSISTED_DICT_VERSION + 1, "entries": {"a": "1"}})
    )
    d = PersistedDict(path=_path(tmp_path))
    assert d.get("a") is None
    assert len(d) == 0


def test_an_unreadable_file_starts_empty(tmp_path):
    _path(tmp_path).write_text("{not json")
    assert len(PersistedDict(path=_path(tmp_path))) == 0


def test_a_reset_cache_can_be_written_again(tmp_path):
    _path(tmp_path).write_text(
        json.dumps({"version": PERSISTED_DICT_VERSION + 1, "entries": {"a": "1"}})
    )
    d = PersistedDict(path=_path(tmp_path))
    d["b"] = "2"
    data = json.loads(_path(tmp_path).read_text())
    assert data["version"] == PERSISTED_DICT_VERSION
    assert data["entries"] == {"b": "2"}
