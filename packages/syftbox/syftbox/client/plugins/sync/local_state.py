import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import Self, Type

from syftbox.client.base import SyftBoxContextInterface
from syftbox.client.plugins.sync.exceptions import SyncEnvironmentError
from syftbox.client.plugins.sync.sync_action import SyncAction
from syftbox.client.plugins.sync.types import SyncActionType, SyncStatus
from syftbox.server.models.sync_models import FileMetadata

LOCAL_STATE_FILENAME = "local_syncstate.json"


class SyncStatusInfo(BaseModel):
    path: Path
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    status: SyncStatus
    message: Optional[str] = None
    action: Optional[SyncActionType] = None


class LocalState(BaseModel):
    path: Path = Field(description="Path to the LocalState file")
    # The state of files on last successful sync
    states: dict[Path, FileMetadata] = {}
    # The last sync status of each file
    status_info: dict[Path, SyncStatusInfo] = {}

    @classmethod
    def for_context(cls: Type[Self], context: SyftBoxContextInterface) -> Self:
        return cls(path=context.workspace.plugins / LOCAL_STATE_FILENAME)

    def insert_completed_action(self, action: SyncAction) -> None:
        """Insert action result into local state."""
        if action.action_type == SyncActionType.NOOP:
            return

        if action.status == SyncStatus.PROCESSING:
            raise ValueError("Attempted to insert an action into LocalState that is still being processed.")

        if action.status == SyncStatus.SYNCED:
            self.insert_synced_file(
                path=action.path,
                state=action.result_local_state,
                action=action.action_type,
            )
        else:
            self.insert_status_info(
                path=action.path,
                status=action.status,
                message=action.message,
                action=action.action_type,
            )

    def insert_synced_file(
        self, path: Path, state: Optional[FileMetadata], action: "SyncActionType", save: bool = True
    ) -> None:
        if not isinstance(path, Path):
            raise ValueError(f"path must be a Path object, got {path}")
        if not self.path.is_file():
            # If the LocalState file does not exist, the sync environment is corrupted and syncing should be aborted

            # NOTE: this can occur when the user deletes the sync folder, but a different plugin re-creates it.
            # If the sync folder exists but the LocalState file does not, it means the sync folder was deleted
            # during syncing and might cause unexpected behavior like deleting files on the remote
            raise SyncEnvironmentError("Your previous sync state has been deleted by a different process.")

        if state is None:
            self.states.pop(path, None)
        else:
            self.states[path] = state

        self.insert_status_info(
            path,
            SyncStatus.SYNCED,
            action=action,
            save=False,
        )
        if save:
            self.save()

    def insert_status_info(
        self,
        path: Path,
        status: SyncStatus,
        message: Optional[str] = None,
        action: Optional["SyncActionType"] = None,
        save: bool = True,
    ) -> None:
        if not isinstance(path, Path):
            raise ValueError(f"path must be a Path object, got {path}")
        self.status_info[path] = SyncStatusInfo(path=path, status=status, message=message, action=action)
        if save:
            self.save()

    def save(self) -> None:
        try:
            with threading.Lock():
                self.path.write_text(self.model_dump_json())
        except Exception as e:
            logger.exception(f"Failed to save {self.path}: {e}")

    def load(self) -> None:
        with threading.Lock():
            if self.path.exists():
                data = self.path.read_text()
                loaded_state = self.model_validate_json(data)
                self.states = loaded_state.states
                self.status_info = loaded_state.status_info
            else:
                # Ensure the file exists for the next save
                self.save()
