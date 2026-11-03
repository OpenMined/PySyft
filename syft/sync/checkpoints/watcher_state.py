"""What a DS received from each DO, kept in the DS's own storage.

A lost local disk, as after a Colab runtime reset, loses the files from a DO
and the cursor in its outbox. The DS restores both from the checkpoint and the
rolling state of that DO, so it does not apply an outbox message again.
"""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict

from syft.sync.events.file_change_event import FileChangeEvent
from syft.sync.utils.syftbox_utils import compress_data, uncompress_data
from syft.sync.utils.versioned_file import parse_versioned_json

# Format of the checkpoint and the rolling state. Raise it when the layout of
# either changes. A state of a later version is refused.
WATCHER_STATE_VERSION = 1


class _WatcherState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Names the state in errors.
    kind: ClassVar[str]

    version: int

    def as_compressed_data(self) -> bytes:
        return compress_data(self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_compressed_data(cls, data: bytes) -> Self:
        """Raises:
        ValueError: if a later client wrote the state, or it has another shape.
        """
        return parse_versioned_json(
            uncompress_data(data), cls, WATCHER_STATE_VERSION, cls.kind
        )


class WatcherCheckpoint(_WatcherState):
    """The files received from one DO: the latest event of each path. A deleted
    path has no event."""

    kind: ClassVar[str] = "watcher checkpoint"

    events: list[FileChangeEvent]


class WatcherRollingState(_WatcherState):
    """The changes received from one DO since its checkpoint, and the cursor in
    its outbox."""

    kind: ClassVar[str] = "watcher rolling state"

    # The latest event of each path since the checkpoint, deletes included.
    events: list[FileChangeEvent]
    # The msg_id of the newest events message received from the DO.
    last_msg_id: str
    # The ids of the events messages received. The outbox can list some of
    # them again, and the DS skips those.
    received_msg_ids: frozenset[str]
