"""The DO's own storage on local disk: event log, checkpoints, rolling state.

A WebRTC user has no Drive, so the DO keeps this state on its own machine. A
new machine starts with no state.
"""

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from syft.sync.checkpoints.checkpoint import Checkpoint, IncrementalCheckpoint
from syft.sync.checkpoints.rolling_state import RollingState
from syft.sync.connections.protocols import OwnerStoreConnection
from syft.sync.events.file_change_event import FileChangeEventsMessageFileName
from syft.sync.sync.constants import CACHE_DIR

# The owner store, under the syftbox folder.
OWNER_STORE_PATH = Path(CACHE_DIR) / "webrtc" / "owner_store"
EVENTS_FOLDER = "events"
CHECKPOINTS_FOLDER = "checkpoints"
INCREMENTALS_FOLDER = "incrementals"
ROLLING_STATE_FOLDER = "rolling_state"
# Each file goes here first, then moves into place. Thus a crash never leaves
# half a file in the store.
TEMP_FOLDER = "tmp"

NameValue = TypeVar("NameValue")


def parse_file_name(parse: Callable[[str], NameValue | None], name: str) -> NameValue:
    """Return the value that ``parse`` reads from the file name ``name``.

    Raises:
        ValueError: ``parse`` finds no value in ``name``.
    """
    value = parse(name)
    if value is None:
        raise ValueError(f"The owner store holds a file with a bad name: {name!r}")
    return value


def event_timestamp(name: str) -> float:
    return FileChangeEventsMessageFileName.from_string(name).timestamp


def incremental_sequence(name: str) -> int:
    return parse_file_name(IncrementalCheckpoint.filename_to_sequence_number, name)


class LocalOwnerStoreConnection(OwnerStoreConnection):
    """``OwnerStoreConnection`` in a folder on local disk.

    A handle is the path of a file, relative to the folder.
    """

    def __init__(self, folder: Path):
        self._folder = folder

    def append_event(self, name: str, data: bytes) -> str:
        return self._write_file(EVENTS_FOLDER, name, data)

    def list_event_handles(self, since_timestamp: float | None = None) -> list[str]:
        names = sorted(self._file_names(EVENTS_FOLDER), key=event_timestamp)
        return [
            f"{EVENTS_FOLDER}/{name}"
            for name in names
            if since_timestamp is None or event_timestamp(name) > since_timestamp
        ]

    def fetch(self, handle: str) -> bytes:
        return (self._folder / handle).read_bytes()

    def put_checkpoint(self, name: str, data: bytes) -> str:
        return self._replace_file(CHECKPOINTS_FOLDER, name, data)

    def latest_checkpoint(self) -> bytes | None:
        return self._read_latest_file(
            CHECKPOINTS_FOLDER, Checkpoint.filename_to_timestamp
        )

    def put_incremental(self, name: str, data: bytes) -> str:
        return self._write_file(INCREMENTALS_FOLDER, name, data)

    def all_incrementals(self) -> list[bytes]:
        names = sorted(self._file_names(INCREMENTALS_FOLDER), key=incremental_sequence)
        return [self._read_file(INCREMENTALS_FOLDER, name) for name in names]

    def incremental_count(self) -> int:
        return len(self._file_names(INCREMENTALS_FOLDER))

    def next_incremental_sequence(self) -> int:
        names = self._file_names(INCREMENTALS_FOLDER)
        return max((incremental_sequence(name) for name in names), default=0) + 1

    def delete_incrementals(self) -> None:
        self._delete_files(INCREMENTALS_FOLDER)

    def put_rolling_state(self, name: str, data: bytes) -> str:
        return self._replace_file(ROLLING_STATE_FOLDER, name, data)

    def get_rolling_state(self) -> bytes | None:
        return self._read_latest_file(
            ROLLING_STATE_FOLDER, RollingState.filename_to_timestamp
        )

    def delete_rolling_state(self) -> None:
        self._delete_files(ROLLING_STATE_FOLDER)

    def delete_all(self) -> int:
        """Delete the folder of the store and everything in it.

        Returns:
            The number of files and folders that went.
        """
        if not self._folder.exists():
            return 0
        # Reverse order puts each path before the folder that holds it.
        paths = sorted(self._folder.rglob("*"), reverse=True) + [self._folder]
        for path in paths:
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        return len(paths)

    def _file_names(self, folder: str) -> list[str]:
        path = self._folder / folder
        if not path.exists():
            return []
        return [file.name for file in path.iterdir()]

    def _read_file(self, folder: str, name: str) -> bytes:
        return (self._folder / folder / name).read_bytes()

    def _write_file(self, folder: str, name: str, data: bytes) -> str:
        """Write ``data`` to the file ``name`` in ``folder``, and return its
        handle."""
        temp = self._folder / TEMP_FOLDER / uuid.uuid4().hex
        target = self._folder / folder / name
        temp.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(data)
        os.replace(temp, target)
        return f"{folder}/{name}"

    def _replace_file(self, folder: str, name: str, data: bytes) -> str:
        """Write the file ``name`` in ``folder``, then delete the other files in
        ``folder``."""
        handle = self._write_file(folder, name, data)
        for old_name in self._file_names(folder):
            if old_name != name:
                (self._folder / folder / old_name).unlink()
        return handle

    def _read_latest_file(
        self, folder: str, timestamp_of: Callable[[str], float | None]
    ) -> bytes | None:
        names = self._file_names(folder)
        if not names:
            return None
        latest = max(names, key=lambda name: parse_file_name(timestamp_of, name))
        return self._read_file(folder, latest)

    def _delete_files(self, folder: str) -> None:
        for name in self._file_names(folder):
            (self._folder / folder / name).unlink()
