"""Shared exclusive file-locking helper for read-modify-write cycles."""

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on lock_path for the block's duration."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    with open(lock_path) as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
