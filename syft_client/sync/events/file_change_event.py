from typing import Any, List, Literal
from pathlib import Path
from uuid import UUID, uuid4
import base64
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_serializer,
    computed_field,
)
from syft_client.sync.messages.proposed_filechange import ProposedFileChange
from syft_client.sync.utils.syftbox_utils import create_event_timestamp
from syft_client.sync.utils.syftbox_utils import compress_data
from syft_client.sync.utils.syftbox_utils import uncompress_data


FILE_CHANGE_FILENAME_PREFIX = "syfteventsmessagev3"
DEFAULT_EVENT_FILENAME_EXTENSION = ".tar.gz"


class FileChangeEventsMessageFileName(BaseModel):
    id: UUID = Field(default_factory=lambda: uuid4())
    timestamp: float = Field(default_factory=lambda: create_event_timestamp())
    extension: str = DEFAULT_EVENT_FILENAME_EXTENSION

    def as_string(self) -> str:
        return f"{FILE_CHANGE_FILENAME_PREFIX}_{self.timestamp}_{self.id}{DEFAULT_EVENT_FILENAME_EXTENSION}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FileChangeEventsMessageFileName):
            return False
        return (
            self.id == other.id
            and self.timestamp == other.timestamp
            and self.extension == other.extension
        )

    @classmethod
    def from_string(cls, filename: str) -> "FileChangeEventsMessageFileName":
        try:
            parts = filename.split("_", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid filename: {filename}")
            timestamp = float(parts[1])

            id_with_ext = parts[2]
            _id = id_with_ext
            if _id.endswith(DEFAULT_EVENT_FILENAME_EXTENSION):
                _id = UUID(_id[: -len(DEFAULT_EVENT_FILENAME_EXTENSION)])
            return cls(id=_id, timestamp=timestamp)
        except Exception as e:
            raise ValueError(f"Invalid filename: {filename}") from e


class FileChangeEvent(BaseModel):
    id: UUID
    path_in_datasite: Path
    datasite_email: str
    content: str | bytes | None = (
        None  # None for deletions, can be str or bytes for binary files
    )
    old_hash: str | None = None
    new_hash: str | None = None  # None for deletions
    is_deleted: bool = False
    submitted_timestamp: float
    timestamp: float

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

    @property
    def path_in_syftbox(self) -> Path:
        return Path(self.datasite_email) / self.path_in_datasite

    @model_validator(mode="before")
    @classmethod
    def pre_init(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Deserialize content based on content_type metadata
        content_type = data.get("content_type", None)
        content = data.get("content")

        if content is not None and isinstance(content, str):
            if content_type == "binary":
                # Definitively decode base64 since we know it was binary
                data["content"] = base64.b64decode(content)
            # If content_type is "text" or None (legacy), keep as string

        return data

    def eventfile_filepath(self) -> str:
        # TODO: remove
        return f"_{self.id}"

    @classmethod
    def from_proposed_filechange(
        cls,
        proposed_filechange: ProposedFileChange,
    ) -> "FileChangeEvent":
        return cls(
            path_in_datasite=proposed_filechange.path_in_datasite,
            content=proposed_filechange.content,
            id=proposed_filechange.id,
            old_hash=proposed_filechange.old_hash,
            new_hash=proposed_filechange.new_hash,
            submitted_timestamp=proposed_filechange.submitted_timestamp,
            timestamp=create_event_timestamp(),
            datasite_email=proposed_filechange.datasite_email,
        )

    def __hash__(self) -> int:
        # this is for comparing locally
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FileChangeEvent):
            return False
        return self.id == other.id


class FileChangeEventsMessage(BaseModel):
    events: List[FileChangeEvent]
    message_filepath: FileChangeEventsMessageFileName = Field(
        default_factory=lambda: FileChangeEventsMessageFileName()
    )

    @property
    def timestamp(self) -> float:
        return self.message_filepath.timestamp

    def as_compressed_data(self) -> bytes:
        return compress_data(self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_compressed_data(cls, data: bytes) -> "FileChangeEvent":
        uncompressed_data = uncompress_data(data)
        return cls.model_validate_json(uncompressed_data)
