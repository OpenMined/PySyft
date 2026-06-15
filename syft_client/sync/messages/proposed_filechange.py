from typing import List, Any, Literal
from uuid import UUID, uuid4
from pathlib import Path
import uuid
import time
import base64
from pydantic import Field, model_validator, field_serializer, computed_field
from pydantic.main import BaseModel
from syft_client.sync.utils.syftbox_utils import compress_data, uncompress_data
from syft_client.sync.utils.syftbox_utils import create_event_timestamp
from syft_client.sync.utils.syftbox_utils import get_event_hash_from_content


MESSAGE_FILENAME_PREFIX = "msgv2"
MESSAGE_FILENAME_EXTENSION = ".tar.gz"


class ProposedFileChange(BaseModel):
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
        if self.content is None:
            return None
        elif isinstance(self.content, bytes):
            return "binary"
        else:
            return "text"

    @field_serializer("content", when_used="json")
    def serialize_content(self, value: str | bytes | None) -> str | None:
        """Serialize bytes as base64-encoded string for JSON."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("utf-8")
        return value

    @model_validator(mode="before")
    @classmethod
    def pre_init(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Deserialize content based on content_type metadata
        content_type = data.get("content_type")
        content = data.get("content")

        if content is not None and isinstance(content, str):
            if content_type == "binary":
                # Definitively decode base64 since we know it was binary
                data["content"] = base64.b64decode(content)
            # If content_type is "text" or None (legacy), keep as string

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


class ProposedFileChangesMessage(BaseModel):
    id: UUID = Field(default_factory=lambda: uuid4())
    sender_email: str
    message_filename: MessageFileName = Field(default_factory=lambda: MessageFileName())
    proposed_file_changes: List[ProposedFileChange]
    # Platform-specific ID (e.g., Google Drive file ID) - set when retrieving message
    # Used to avoid re-querying the platform when removing the message
    platform_id: str | None = Field(default=None, exclude=True)

    @classmethod
    def from_compressed_data(cls, data: bytes) -> "ProposedFileChangesMessage":
        uncompressed_data = uncompress_data(data)
        return cls.model_validate_json(uncompressed_data)

    def as_compressed_data(self) -> bytes:
        data = self.model_dump_json(indent=2).encode("utf-8")
        return compress_data(data)
