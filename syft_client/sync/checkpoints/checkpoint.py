"""
Checkpoint data models for syft-client.

Two types of checkpoints:
1. IncrementalCheckpoint - stores events for a range (e.g., events 1-50)
2. Checkpoint - full state snapshot (used after compacting)

Flow:
- Events accumulate in RollingState
- At threshold (e.g., 50 events): RollingState → IncrementalCheckpoint
- After N incremental checkpoints: compact into single full Checkpoint
"""

from typing import List, Dict, TYPE_CHECKING
from pydantic import BaseModel, Field
from pathlib import Path
from syft_client.sync.utils.syftbox_utils import (
    create_event_timestamp,
    compress_data,
    uncompress_data,
)

if TYPE_CHECKING:
    from syft_client.sync.events.file_change_event import FileChangeEvent


CHECKPOINT_FILENAME_PREFIX = "checkpoint"
INCREMENTAL_CHECKPOINT_PREFIX = "incremental_checkpoint"
CHECKPOINT_VERSION = 1

# Default compacting threshold: merge after this many incremental checkpoints
DEFAULT_COMPACTING_THRESHOLD = 4


class CheckpointFile(BaseModel):
    """Represents a single file in the checkpoint."""

    path: str  # Relative path in datasite (e.g., "public/syft_datasets/data.csv")
    hash: str  # Hash of the content (SHA256 hex string)
    content: str  # File content


class Checkpoint(BaseModel):
    """
    A checkpoint is a complete snapshot of the current state.

    Contains all files and their hashes at a specific point in time.
    Used to speed up initial sync by avoiding downloading all historical events.
    """

    version: int = CHECKPOINT_VERSION
    timestamp: float = Field(default_factory=create_event_timestamp)
    email: str
    last_event_timestamp: float | None = None  # Timestamp of last event included
    files: List[CheckpointFile] = []

    @property
    def filename(self) -> str:
        """Generate filename for this checkpoint."""
        return f"{CHECKPOINT_FILENAME_PREFIX}_{self.timestamp}.tar.gz"

    @classmethod
    def filename_to_timestamp(cls, filename: str) -> float | None:
        """Extract timestamp from checkpoint filename."""
        try:
            # Format: checkpoint_<timestamp>.tar.gz
            if not filename.startswith(CHECKPOINT_FILENAME_PREFIX):
                return None
            parts = filename.replace(".tar.gz", "").split("_")
            if len(parts) >= 2:
                return float(parts[1])
            return None
        except (ValueError, IndexError):
            return None

    @classmethod
    def from_file_hashes_and_contents(
        cls,
        email: str,
        file_hashes: Dict[str, str],
        file_contents: Dict[str, str],
        last_event_timestamp: float | None = None,
    ) -> "Checkpoint":
        """
        Create a checkpoint from file_hashes dict and file contents.

        Args:
            email: Data owner email
            file_hashes: Dict mapping path -> hash
            file_contents: Dict mapping path -> content
            last_event_timestamp: Timestamp of last event included in this checkpoint
        """
        files = []
        for path, hash_value in file_hashes.items():
            path_str = str(path)
            content = file_contents.get(path_str) or file_contents.get(path)
            if content is not None:
                files.append(
                    CheckpointFile(
                        path=path_str,
                        hash=hash_value,
                        content=content,
                    )
                )
        return cls(
            email=email,
            files=files,
            last_event_timestamp=last_event_timestamp,
        )

    def get_file_hashes(self) -> Dict[Path, str]:
        """Get file hashes as a dict mapping path to hash."""
        return {Path(f.path): f.hash for f in self.files}

    def get_file_contents(self) -> Dict[str, str]:
        """Get file contents as a dict mapping path to content."""
        return {f.path: f.content for f in self.files}

    def as_compressed_data(self) -> bytes:
        """Compress checkpoint for storage."""
        return compress_data(self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_compressed_data(cls, data: bytes) -> "Checkpoint":
        """Load checkpoint from compressed data."""
        uncompressed_data = uncompress_data(data)
        return cls.model_validate_json(uncompressed_data)


class IncrementalCheckpoint(BaseModel):
    """
    An incremental checkpoint stores events for a specific range.

    Unlike a full Checkpoint (which stores complete file state), an incremental
    checkpoint stores FileChangeEvent objects - the same structure as RollingState.
    This enables efficient storage and simple compacting.

    Example: If checkpoint threshold is 50:
    - IncrementalCheckpoint #1: events 1-50
    - IncrementalCheckpoint #2: events 51-100
    - After compacting: single full Checkpoint with merged state
    """

    version: int = CHECKPOINT_VERSION
    timestamp: float = Field(default_factory=create_event_timestamp)
    email: str
    sequence_number: int  # 1, 2, 3, ... (which checkpoint this is)
    events: List["FileChangeEvent"] = Field(default_factory=list)

    @property
    def filename(self) -> str:
        """Generate filename: incremental_checkpoint_{seq}_{timestamp}.tar.gz"""
        return f"{INCREMENTAL_CHECKPOINT_PREFIX}_{self.sequence_number}_{self.timestamp}.tar.gz"

    @property
    def event_count(self) -> int:
        """Number of events in this checkpoint."""
        return len(self.events)

    @classmethod
    def filename_to_sequence_number(cls, filename: str) -> int | None:
        """Extract sequence number from filename."""
        try:
            if not filename.startswith(INCREMENTAL_CHECKPOINT_PREFIX):
                return None
            # Format: incremental_checkpoint_{seq}_{timestamp}.tar.gz
            parts = filename.replace(".tar.gz", "").split("_")
            # parts = ["incremental", "checkpoint", "{seq}", "{timestamp}"]
            if len(parts) >= 4:
                return int(parts[2])
            return None
        except (ValueError, IndexError):
            return None

    def as_compressed_data(self) -> bytes:
        """Compress for storage."""
        return compress_data(self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_compressed_data(cls, data: bytes) -> "IncrementalCheckpoint":
        """Load from compressed data."""
        uncompressed_data = uncompress_data(data)
        return cls.model_validate_json(uncompressed_data)


def compact_incremental_checkpoints(
    checkpoints: List[IncrementalCheckpoint],
    email: str,
) -> Checkpoint:
    """
    Merge multiple incremental checkpoints into a single full Checkpoint.

    Events are merged in order (by sequence number), with later events
    overwriting earlier ones for the same file path. Deleted files are excluded.

    Args:
        checkpoints: List of incremental checkpoints to merge.
        email: Email for the resulting checkpoint.

    Returns:
        A full Checkpoint with the merged state.
    """
    # Sort by sequence number to process in order
    sorted_checkpoints = sorted(checkpoints, key=lambda c: c.sequence_number)

    # Merge events: later events overwrite earlier ones for same path
    merged_events: Dict[str, "FileChangeEvent"] = {}
    last_event_timestamp = None

    for cp in sorted_checkpoints:
        for event in cp.events:
            merged_events[str(event.path_in_datasite)] = event
            if event.timestamp is not None:
                if (
                    last_event_timestamp is None
                    or event.timestamp > last_event_timestamp
                ):
                    last_event_timestamp = event.timestamp

    # Convert to full Checkpoint (only non-deleted files)
    files = []
    for event in merged_events.values():
        if not event.is_deleted and event.content is not None:
            files.append(
                CheckpointFile(
                    path=str(event.path_in_datasite),
                    hash=str(event.new_hash),
                    content=event.content,
                )
            )

    return Checkpoint(
        email=email,
        files=files,
        last_event_timestamp=last_event_timestamp,
    )


# Import here to avoid circular imports
from syft_client.sync.events.file_change_event import FileChangeEvent  # noqa: E402

# Update forward references
IncrementalCheckpoint.model_rebuild()
