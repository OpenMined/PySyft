"""A dict subclass that transparently persists to a JSON file.

Reads always reload from disk (sees other process's writes). Writes always
save to disk immediately. Cross-process consistency is enforced by a
portalocker file lock: reads take a shared lock, writes take an exclusive
lock, and read-modify-write operations hold the exclusive lock for the whole
cycle so updates are not lost.

Usage:
    file_hashes = PersistedDict(
        path=syftbox_folder / ".cache" / "owner_file_hashes.json",
        key_serializer=str,           # convert keys to JSON-safe form
        key_deserializer=Path,        # convert back on load
    )
    file_hashes["foo.txt"] = "hash"   # auto-saved under exclusive lock
    file_hashes.get("foo.txt")        # reads latest under shared lock

Batch writes (avoid one rename per item):
    with file_hashes.exclusive_lock():
        for k, v in many_items:
            file_hashes.set(k, v, write=False)
        file_hashes._write_to_file()
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import portalocker


class PersistedDict(dict):
    """Dict that persists to a JSON file. With path=None it's a plain in-memory dict."""

    def __init__(
        self,
        path: Path | None = None,
        key_serializer: Callable[[Any], str] = str,
        key_deserializer: Callable[[str], Any] = lambda s: s,
    ):
        super().__init__()
        self._path = path
        self._key_serializer = key_serializer
        self._key_deserializer = key_deserializer
        # Load existing state on construction (no-op if path is None)
        self._load_from_disk()

    # --- locking primitives ---

    def _lock_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".lock")

    @contextmanager
    def _locked(self, exclusive: bool):
        if self._path is None:
            yield
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        flag = portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH
        with open(self._lock_path(), "a") as f:
            portalocker.lock(f, flag)
            try:
                yield
            finally:
                portalocker.unlock(f)

    def exclusive_lock(self):
        """Acquire a cross-process exclusive lock on this dict's file.

        Inside this block, no other process can read or write the dict via
        the locked APIs. Use it to batch multiple `set(..., write=False)`
        calls followed by a single `_write_to_file`.
        """
        return self._locked(exclusive=True)

    def shared_lock(self):
        """Acquire a cross-process shared lock on this dict's file.

        Multiple readers can hold the shared lock simultaneously, but no
        exclusive writer can proceed while any shared lock is held.
        """
        return self._locked(exclusive=False)

    # --- I/O without locking (caller must hold the appropriate lock) ---

    def _read_from_file(self) -> None:
        if self._path is None:
            return  # in-memory mode: never clear on read
        super().clear()
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for k, v in data.items():
                super().__setitem__(self._key_deserializer(k), v)
        except (json.JSONDecodeError, OSError):
            pass

    def _write_to_file(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process unique tmp path: even with the file lock, this guards
        # against any path where two writers share a tmp filename.
        tmp = self._path.with_suffix(f".tmp.{os.getpid()}.{uuid4().hex}")
        serialized = {self._key_serializer(k): v for k, v in super().items()}
        try:
            tmp.write_text(json.dumps(serialized))
            tmp.replace(self._path)
        finally:
            tmp.unlink(missing_ok=True)

    # --- I/O with locking ---

    def _load_from_disk(self) -> None:
        if self._path is None:
            return  # in-memory mode
        with self.shared_lock():
            self._read_from_file()

    def _save_to_disk(self) -> None:
        if self._path is None:
            return  # in-memory mode
        with self.exclusive_lock():
            self._write_to_file()

    # --- explicit-control mutation helpers ---
    #
    # `read=False` skips the disk reload (use when caller already holds the
    # exclusive lock so in-memory state is current).
    # `write=False` skips persistence (use to batch many mutations under one
    # exclusive_lock and call `_write_to_file` once at the end).

    def set(self, key: Any, value: Any, write: bool = True) -> None:
        if write:
            self[key] = value
        else:
            super().__setitem__(key, value)

    def delete(self, key: Any, write: bool = True) -> None:
        if write:
            del self[key]
        else:
            super().__delitem__(key)

    def contains(self, key: Any, read: bool = True) -> bool:
        if read:
            return key in self
        return super().__contains__(key)

    # --- read overrides: shared lock for the whole read ---

    def __getitem__(self, key):
        with self.shared_lock():
            self._read_from_file()
            return super().__getitem__(key)

    def __contains__(self, key):
        with self.shared_lock():
            self._read_from_file()
            return super().__contains__(key)

    def __iter__(self):
        with self.shared_lock():
            self._read_from_file()
            # Materialize while still holding the lock so iteration is over a
            # stable snapshot.
            return iter(list(super().__iter__()))

    def __len__(self):
        with self.shared_lock():
            self._read_from_file()
            return super().__len__()

    def get(self, key, default=None):
        with self.shared_lock():
            self._read_from_file()
            return super().get(key, default)

    def keys(self):
        with self.shared_lock():
            self._read_from_file()
            return list(super().keys())

    def values(self):
        with self.shared_lock():
            self._read_from_file()
            return list(super().values())

    def items(self):
        with self.shared_lock():
            self._read_from_file()
            return list(super().items())

    # --- write overrides: single exclusive lock around read-modify-write ---

    def __setitem__(self, key, value):
        with self.exclusive_lock():
            self._read_from_file()
            super().__setitem__(key, value)
            self._write_to_file()

    def __delitem__(self, key):
        with self.exclusive_lock():
            self._read_from_file()
            super().__delitem__(key)
            self._write_to_file()

    def pop(self, key, *args):
        with self.exclusive_lock():
            self._read_from_file()
            result = super().pop(key, *args)
            self._write_to_file()
            return result

    def update(self, *args, **kwargs):
        with self.exclusive_lock():
            self._read_from_file()
            super().update(*args, **kwargs)
            self._write_to_file()

    def clear(self):
        with self.exclusive_lock():
            super().clear()
            self._write_to_file()
