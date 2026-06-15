"""
Rolling state data models for syft-client.

A rolling state keeps the latest state of each file since the last checkpoint,
enabling fast sync by downloading checkpoint + rolling_state instead of many
individual event files.

Flow:
1. Checkpoint created at event N (threshold reached)
2. Events N+1 to M update rolling state (keeping only latest per file)
3. On fresh login: download checkpoint + rolling_state = 2 API calls
4. When new checkpoint is created, rolling state is deleted and reset
"""

from typing import List
from pydantic import BaseModel, Field
from syft_client.sync.utils.syftbox_utils import (
    create_event_timestamp,
    compress_data,
    uncompress_data,
)
from syft_client.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
)


ROLLING_STATE_FILENAME_PREFIX = "rolling_state"
ROLLING_STATE_VERSION = 1


class RollingState(BaseModel):
    """
    Rolling state keeps the latest state of each file since the last checkpoint.

    This provides an optimization for initial sync:
    - Without rolling state: download checkpoint + N individual event files
    - With rolling state: download checkpoint + 1 rolling state file

    The rolling state is updated after each event and deleted when a new
    checkpoint is created.
    """

    version: int = ROLLING_STATE_VERSION
    timestamp: float = Field(default_factory=create_event_timestamp)
    email: str
    base_checkpoint_timestamp: float  # Timestamp of the checkpoint this builds on
    last_event_timestamp: float | None = None  # Timestamp of last event included
    events: List[FileChangeEvent] = Field(default_factory=list)

    @property
    def filename(self) -> str:
        """Generate filename for this rolling state."""
        return f"{ROLLING_STATE_FILENAME_PREFIX}_{self.timestamp}.tar.gz"

    @property
    def event_count(self) -> int:
        """Number of events in this rolling state."""
        return len(self.events)

    def add_event(self, event: FileChangeEvent) -> None:
        """
        Add a single event to the rolling state.

        Only keeps the latest event per file path. If an event for the same
        path already exists, it is replaced with the new event.
        """
        # Remove any existing event for the same file path
        self.events = [
            e for e in self.events if e.path_in_datasite != event.path_in_datasite
        ]
        self.events.append(event)
        self.last_event_timestamp = event.timestamp
        self.timestamp = create_event_timestamp()

    def add_events_message(self, events_message: FileChangeEventsMessage) -> None:
        """
        Add all events from a message to the rolling state.

        Only keeps the latest event per file path. Events within the message
        are processed in order, and later events for the same path replace
        earlier ones.
        """
        for event in events_message.events:
            # Remove any existing event for the same file path
            self.events = [
                e for e in self.events if e.path_in_datasite != event.path_in_datasite
            ]
            self.events.append(event)
        if events_message.events:
            self.last_event_timestamp = max(e.timestamp for e in events_message.events)
        self.timestamp = create_event_timestamp()

    def clear(self, new_base_checkpoint_timestamp: float) -> None:
        """
        Clear events and reset for a new checkpoint.

        Args:
            new_base_checkpoint_timestamp: Timestamp of the new checkpoint
        """
        self.events = []
        self.base_checkpoint_timestamp = new_base_checkpoint_timestamp
        self.last_event_timestamp = None
        self.timestamp = create_event_timestamp()

    def as_compressed_data(self) -> bytes:
        """Compress rolling state for storage."""
        return compress_data(self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_compressed_data(cls, data: bytes) -> "RollingState":
        """Load rolling state from compressed data."""
        uncompressed_data = uncompress_data(data)
        return cls.model_validate_json(uncompressed_data)

    @classmethod
    def filename_to_timestamp(cls, filename: str) -> float | None:
        """Extract timestamp from rolling state filename."""
        try:
            # Format: rolling_state_<timestamp>.tar.gz
            if not filename.startswith(ROLLING_STATE_FILENAME_PREFIX):
                return None
            parts = filename.replace(".tar.gz", "").split("_")
            # rolling_state_<timestamp> -> ["rolling", "state", "<timestamp>"]
            if len(parts) >= 3:
                return float(parts[2])
            return None
        except (ValueError, IndexError):
            return None
