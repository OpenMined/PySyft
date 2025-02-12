import base64
import enum
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import AfterValidator, BaseModel, Field


def should_be_relative(v: Path) -> Path:
    if v.is_absolute():
        raise ValueError("path must be relative")
    return v


def should_be_absolute(v: Path) -> Path:
    if not v.is_absolute():
        raise ValueError("path must be absolute")
    return v


RelativePath = Annotated[Path, AfterValidator(should_be_relative)]

AbsolutePath = Annotated[Path, AfterValidator(should_be_absolute)]


class DiffRequest(BaseModel):
    path: RelativePath
    signature: str

    @property
    def signature_bytes(self) -> bytes:
        return base64.b85decode(self.signature)


class DiffResponse(BaseModel):
    path: RelativePath
    diff: str
    hash: str

    @property
    def diff_bytes(self) -> bytes:
        return base64.b85decode(self.diff)


class SignatureError(str, enum.Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_NOT_WRITEABLE = "FILE_NOT_WRITEABLE"
    FILE_NOT_READABLE = "FILE_NOT_READABLE"
    NOT_A_FILE = "NOT_A_FILE"


class SignatureResponse(BaseModel):
    path: RelativePath
    signature: Optional[str] = None
    error: Optional[SignatureError] = None


class FileMetadataRequest(BaseModel):
    path: RelativePath = Field(description="Path to search for files")


class FileRequest(BaseModel):
    path: RelativePath = Field(description="Path to search for files, uses SQL LIKE syntax")


class BatchFileRequest(BaseModel):
    paths: list[RelativePath]


class ApplyDiffRequest(BaseModel):
    path: RelativePath
    diff: str
    expected_hash: str

    @property
    def diff_bytes(self) -> bytes:
        return base64.b85decode(self.diff)


class ApplyDiffResponse(BaseModel):
    path: RelativePath
    current_hash: str
    previous_hash: str


class FileMetadata(BaseModel):
    path: Path
    hash: str
    signature: str
    file_size: int = 0
    last_modified: datetime

    @property
    def datasite(self) -> str:
        return self.path.parts[0]

    @staticmethod
    def from_row(row: sqlite3.Row) -> "FileMetadata":
        return FileMetadata(
            path=Path(row["path"]),
            hash=row["hash"],
            signature=row["signature"],
            file_size=row["file_size"],
            last_modified=row["last_modified"],
        )

    @property
    def signature_bytes(self) -> bytes:
        return base64.b85decode(self.signature)

    @property
    def hash_bytes(self) -> bytes:
        return base64.b85decode(self.hash)

    @property
    def datasite_name(self) -> str:
        return self.path.parts[0]

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, FileMetadata):
            return False
        return self.path == value.path and self.hash == value.hash


class SyncLog(BaseModel):
    path: Path
    method: str  # pull or push
    status: str  # success or failure
    timestamp: datetime
    requesting_user: str
