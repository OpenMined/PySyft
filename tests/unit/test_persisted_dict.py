"""Tests for PersistedDict concurrent-write safety and locking semantics."""

import json
import threading
from pathlib import Path

from syft_client.sync.sync.caches.persisted_dict import PersistedDict


def test_concurrent_writes_no_rename_race(tmp_path: Path):
    """Two PersistedDict instances writing to the same path concurrently
    must never raise FileNotFoundError from the atomic rename, and must
    not lose any keys (the exclusive lock serializes the read-modify-write).
    """
    target = tmp_path / "shared.json"
    a = PersistedDict(path=target)
    b = PersistedDict(path=target)

    errors: list[BaseException] = []
    iterations = 100

    def writer(d: PersistedDict, prefix: str):
        try:
            for i in range(iterations):
                d[f"{prefix}-{i}"] = i
        except BaseException as e:
            errors.append(e)

    t1 = threading.Thread(target=writer, args=(a, "a"))
    t2 = threading.Thread(target=writer, args=(b, "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == [], f"Concurrent writes raised: {errors!r}"

    # Every key from both writers must be present in the final on-disk state.
    final = json.loads(target.read_text())
    expected = {f"a-{i}": i for i in range(iterations)} | {
        f"b-{i}": i for i in range(iterations)
    }
    assert final == expected

    # No leftover .tmp files.
    leftover = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftover == [], f"Stale tmp files: {leftover}"


def test_set_with_write_false_does_not_persist(tmp_path: Path):
    """set(..., write=False) mutates memory but leaves disk untouched until
    the caller explicitly persists.
    """
    target = tmp_path / "batch.json"
    d = PersistedDict(path=target)

    d.set("k", "v", write=False)
    assert dict.__getitem__(d, "k") == "v"
    assert not target.exists()

    with d.exclusive_lock():
        d._write_to_file()
    assert json.loads(target.read_text()) == {"k": "v"}


def test_batch_write_with_exclusive_lock(tmp_path: Path):
    """The documented batch pattern: hold the exclusive lock, mutate via
    write=False, persist once at the end. Concurrent batch writers must
    serialize and produce a consistent merged state.
    """
    target = tmp_path / "batch_concurrent.json"
    a = PersistedDict(path=target)
    b = PersistedDict(path=target)

    def batch_write(d: PersistedDict, prefix: str, n: int):
        with d.exclusive_lock():
            d._read_from_file()
            for i in range(n):
                d.set(f"{prefix}-{i}", i, write=False)
            d._write_to_file()

    t1 = threading.Thread(target=batch_write, args=(a, "a", 50))
    t2 = threading.Thread(target=batch_write, args=(b, "b", 50))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    final = json.loads(target.read_text())
    expected = {f"a-{i}": i for i in range(50)} | {f"b-{i}": i for i in range(50)}
    assert final == expected


def test_contains_and_delete_with_flags(tmp_path: Path):
    """contains(read=False) and delete(write=False) bypass the lock-acquiring
    overrides; with default flags they behave like the dunder ops."""
    target = tmp_path / "flags.json"
    d = PersistedDict(path=target)

    d.set("k", "v")  # default write=True
    assert d.contains("k")  # default read=True
    assert d.contains("k", read=False)

    with d.exclusive_lock():
        assert d.contains("k", read=False)
        d.delete("k", write=False)
        assert not d.contains("k", read=False)
        d._write_to_file()

    # After the batch, on-disk state reflects the in-memory delete.
    assert json.loads(target.read_text()) == {}
