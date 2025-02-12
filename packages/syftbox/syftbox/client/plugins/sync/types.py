from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from syftbox.lib.permissions import SyftPermission

# TODO cleanup duplicate types


class SyncStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SYNCED = "synced"
    ERROR = "error"
    REJECTED = "rejected"
    IGNORED = "ignored"


class SyncSide(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class FileChangeInfo(BaseModel):
    local_sync_folder: Path
    path: Path
    side_last_modified: SyncSide
    date_last_modified: datetime
    file_size: int = 1

    @property
    def local_abs_path(self) -> Path:
        return self.local_sync_folder / self.path

    def get_priority(self) -> int:
        if SyftPermission.is_permission_file(self.path):
            return 0
        else:
            return max(1, self.file_size)

    def __lt__(self, other: "FileChangeInfo") -> bool:
        return self.path < other.path


class SyncActionType(str, Enum):
    NOOP = "NOOP"
    CREATE_REMOTE = "CREATE_REMOTE"
    CREATE_LOCAL = "CREATE_LOCAL"
    DELETE_REMOTE = "DELETE_REMOTE"
    DELETE_LOCAL = "DELETE_LOCAL"
    MODIFY_REMOTE = "MODIFY_REMOTE"
    MODIFY_LOCAL = "MODIFY_LOCAL"

    @property
    def target_side(self) -> SyncSide:
        if self in [
            SyncActionType.CREATE_REMOTE,
            SyncActionType.MODIFY_REMOTE,
            SyncActionType.DELETE_REMOTE,
        ]:
            return SyncSide.REMOTE
        return SyncSide.LOCAL
