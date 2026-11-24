from typing import List, Any, Literal
from uuid import UUID, uuid4
from pathlib import Path
import json
import uuid
import time
from pydantic import Field, model_validator, field_serializer, computed_field
from pydantic.main import BaseModel
from syft_migration import MigratableObject

from syft.migrations import client_registry, load_as_latest
from syft.sync.utils.syftbox_utils import compress_data, uncompress_data
from syft.sync.utils.syftbox_utils import create_event_timestamp
from syft.sync.utils.syftbox_utils import get_event_hash_from_content
from syft.sync.utils.file_content import (
    content_to_json,
    content_type_of,
    decode_content,
)


MESSAGE_FILENAME_PREFIX = "msgv2"
MESSAGE_FILENAME_EXTENSION = ".tar.gz"


class ProposedFileChangeV1(BaseModel):
    id: UUID = Field(default_factory=lambda: uuid4())
    old_hash: str | None = None
    new_hash: str | None = None  # None for deletions
    # Use UNIX timestamp (seconds since epoch)
    submitted_timestamp: float = Field(default_factory=lambda: create_event_timestamp())
    path_in_datasite: Path
    content: str | bytes | None = (
        None  # None for deletions, can be str or bytes for binary files
    )
    datasite_email: str
    is_deleted: bool = False

    @computed_field
    @property
    def content_type(self) -> Literal["text", "binary"] | None:
        """Computed field that stores the content type for proper deserialization."""
        return content_type_of(self.content)

    @field_serializer("content", when_used="json")
    def serialize_content(self, value: str | bytes | None) -> str | None:
        """Serialize bytes as base64-encoded string for JSON."""
        return content_to_json(value)

    @model_validator(mode="before")
    @classmethod
    def pre_init(cls, data: dict[str, Any]) -> dict[str, Any]:
        data = decode_content(data)

        # Generate hash if needed
        if "new_hash" not in data and not data.get("is_deleted", False):
            content = data.get("content")
            if content is not None:
                data["new_hash"] = get_event_hash_from_content(content)

        return data


class FileNameParseError(Exception):
    pass


class MessageFileName(BaseModel):
    submitted_timestamp: float = Field(default_factory=lambda: time.time())
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def as_string(self) -> str:
        return f"{MESSAGE_FILENAME_PREFIX}_{self.submitted_timestamp}_{self.uid}{MESSAGE_FILENAME_EXTENSION}"

    @classmethod
    def from_string(cls, filename: str) -> "MessageFileName":
        try:
            parts = filename.split("_")
            if len(parts) != 3:
                raise ValueError(f"Invalid filename: {filename}")
            submitted_timestamp = float(parts[1])
            uid = parts[2].replace(MESSAGE_FILENAME_EXTENSION, "")
        except Exception:
            raise FileNameParseError(f"Invalid filename: {filename}")
        return cls(submitted_timestamp=submitted_timestamp, uid=uid)


class ProposedFileChangesMessageV1(MigratableObject, registry=client_registry):
    """The msgv2 wire envelope (DS -> DO). The envelope is the migratable unit;
    its items are pinned to the exact version class, never a floating alias."""

    canonical_name: str = "ProposedFileChangesMessage"
    version: str = "1"

    id: UUID = Field(default_factory=lambda: uuid4())
    sender_email: str
    message_filename: MessageFileName = Field(default_factory=lambda: MessageFileName())
    proposed_file_changes: List[ProposedFileChangeV1]
    # Platform-specific ID (e.g., Google Drive file ID) - set when retrieving message
    # Used to avoid re-querying the platform when removing the message
    platform_id: str | None = Field(default=None, exclude=True)

    @classmethod
    def from_compressed_data(cls, data: bytes) -> "ProposedFileChangesMessage":
        """Decompress and load, upgraded to the latest version.

        Blobs written by protocol-0 clients (<= 0.1.117) predate the identity
        fields; they are all version 1.
        """
        uncompressed_data = uncompress_data(data)
        return load_as_latest(
            json.loads(uncompressed_data), "ProposedFileChangesMessage"
        )

    def as_compressed_data(self) -> bytes:
        data = self.model_dump_json(indent=2).encode("utf-8")
        return compress_data(data)


# Current-version aliases: callers always work with the latest versions.
ProposedFileChange = ProposedFileChangeV1
ProposedFileChangesMessage = ProposedFileChangesMessageV1
