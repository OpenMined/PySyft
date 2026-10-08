"""Cross-process locks and atomic writes for small local JSON files."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import portalocker


@contextmanager
def locked_file(path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock on the file at ``path``.

    The lock is on a sibling ``.lock`` file, so the file itself can be
    replaced while the lock is held.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(f"{path.suffix}.lock"), "a") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        try:
            yield
        finally:
            portalocker.unlock(f)


def write_json_atomically(path: Path, data: Any, indent: int | None = None) -> None:
    """Replace the file at ``path`` with ``data`` as JSON, in one step.

    A write cut short leaves the old file in place. The data goes to a
    temporary file in the same folder, reaches the disk, and then takes the
    name of the file. ``mkstemp`` creates the temporary file readable by this
    user only.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=indent))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
