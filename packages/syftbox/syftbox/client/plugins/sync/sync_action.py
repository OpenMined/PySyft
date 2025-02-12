import hashlib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional

import py_fast_rsync
from loguru import logger

from syftbox.client.base import SyftBoxContextInterface
from syftbox.client.exceptions import SyftPermissionError
from syftbox.client.plugins.sync.constants import MAX_FILE_SIZE_MB
from syftbox.client.plugins.sync.exceptions import SyncValidationError
from syftbox.client.plugins.sync.types import SyncActionType, SyncSide, SyncStatus
from syftbox.lib.constants import REJECTED_FILE_SUFFIX
from syftbox.lib.permissions import SyftPermission
from syftbox.server.models.sync_models import FileMetadata


def determine_sync_action(
    current_local_metadata: Optional[FileMetadata],
    previous_local_metadata: Optional[FileMetadata],
    current_remote_metadata: Optional[FileMetadata],
) -> "SyncAction":
    """
    Determine the action syncing should take based on the local and remote states, and the previous local state.

    Args:
        current_local_metadata (Optional[FileMetadata]): Metadata of the local file, None if it does not exist.
        previous_local_metadata (Optional[FileMetadata]): Metadata of the local file when it was last synced,
            None if it does not exist.
        current_remote_metadata (Optional[FileMetadata]): Metadata of the remote file, None if it does not exist.

    Raises:
        ValueError: If the action cannot be determined.

    Returns:
        SyncAction: The action to take to sync the local and remote states.
    """
    local_modified = current_local_metadata != previous_local_metadata
    remote_modified = previous_local_metadata != current_remote_metadata
    in_sync = current_remote_metadata == current_local_metadata
    conflict = local_modified and remote_modified
    # If the remote is modified, the local should be updated (possible conflicts are overwritten)
    side_to_update = SyncSide.LOCAL if remote_modified else SyncSide.REMOTE

    local_exists = current_local_metadata is not None
    remote_exists = current_remote_metadata is not None
    both_exist = local_exists and remote_exists

    action: SyncAction

    if in_sync:
        action = NoopAction(local_metadata=current_local_metadata, remote_metadata=current_remote_metadata)  # type: ignore[arg-type]

    # Pull changes from remote
    elif side_to_update == SyncSide.LOCAL and not local_exists:
        action = CreateLocalAction(local_metadata=None, remote_metadata=current_remote_metadata)
    elif side_to_update == SyncSide.LOCAL and both_exist:
        action = ModifyLocalAction(local_metadata=current_local_metadata, remote_metadata=current_remote_metadata)
    elif side_to_update == SyncSide.LOCAL and not remote_exists:
        action = DeleteLocalAction(local_metadata=current_local_metadata, remote_metadata=None)

    # Push changes to remote
    elif side_to_update == SyncSide.REMOTE and not remote_exists:
        action = CreateRemoteAction(local_metadata=current_local_metadata, remote_metadata=None)
    elif side_to_update == SyncSide.REMOTE and both_exist:
        action = ModifyRemoteAction(local_metadata=current_local_metadata, remote_metadata=current_remote_metadata)
    elif side_to_update == SyncSide.REMOTE and not local_exists:
        action = DeleteRemoteAction(local_metadata=None, remote_metadata=current_remote_metadata)
    else:
        raise ValueError("Could not determine sync action")

    logger.debug(
        f"path: {action.path}, "
        f"local_modified: {local_modified}, "
        f"remote_modified: {remote_modified}, "
        f"in_sync: {in_sync}, "
        f"conflict: {conflict}, "
        f"action: {action.action_type.name}"
    )
    return action


def format_rejected_path(path: Path) -> Path:
    return path.with_suffix(REJECTED_FILE_SUFFIX + path.suffix)


class SyncAction(ABC):
    action_type: ClassVar[SyncActionType]
    path: Path
    local_metadata: Optional[FileMetadata]
    remote_metadata: Optional[FileMetadata]
    status: SyncStatus
    message: Optional[str]

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "action_type"):
            raise TypeError("SyncAction subclasses must define an action_type")
        return super().__init_subclass__()

    def __init__(self, local_metadata: Optional[FileMetadata], remote_metadata: Optional[FileMetadata]):
        if not local_metadata and not remote_metadata:
            raise ValueError("At least one of local_metadata or remote_metadata must be provided")
        self.local_metadata = local_metadata
        self.remote_metadata = remote_metadata
        self.path = local_metadata.path if local_metadata else remote_metadata.path  # type: ignore
        self.status = SyncStatus.PROCESSING
        self.message = None

    @property
    def side_to_update(self) -> SyncSide:
        return self.action_type.target_side

    def validate(self, context: SyftBoxContextInterface) -> None:
        """Raises a SyncValidationError if the action is invalid."""
        validate_sync_action(context, self)

    def is_valid(self, context: SyftBoxContextInterface) -> bool:
        try:
            self.validate(context)
            return True
        except SyncValidationError:
            return False

    @abstractmethod
    def execute(self, context: SyftBoxContextInterface) -> None:
        pass

    @abstractmethod
    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        pass

    def error(self, exception: Exception) -> None:
        self.status = SyncStatus.ERROR
        self.message = str(exception)

    @property
    def info_message(self) -> str:
        return f"Syncing {self.path} with action {self.action_type.name}"

    def is_noop(self) -> bool:
        return self.action_type == SyncActionType.NOOP

    @property
    def result_local_state(self) -> Optional[FileMetadata]:
        """Metadata of the local file after the action is executed successfully."""
        if self.side_to_update == SyncSide.LOCAL:
            return self.remote_metadata
        return self.local_metadata


class NoopAction(SyncAction):
    action_type = SyncActionType.NOOP

    def __init__(self, local_metadata: FileMetadata, remote_metadata: FileMetadata) -> None:
        super().__init__(local_metadata, remote_metadata)
        # noop actions are already synced
        self.status = SyncStatus.SYNCED

    def execute(self, context: SyftBoxContextInterface) -> None:
        pass

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        pass


class CreateLocalAction(SyncAction):
    action_type = SyncActionType.CREATE_LOCAL

    def execute(self, context: SyftBoxContextInterface) -> None:
        content_bytes = context.client.sync.download(self.path)
        abs_path = context.workspace.datasites / self.path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content_bytes)
        self.status = SyncStatus.SYNCED

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        # Client doesnt have permission, so no new files are downloaded. Action is a noop.
        self.status = SyncStatus.REJECTED
        self.message = reason


class ModifyLocalAction(SyncAction):
    action_type = SyncActionType.MODIFY_LOCAL

    def execute(self, context: SyftBoxContextInterface) -> None:
        if self.local_metadata is None:
            raise ValueError("Local metadata is required for modify local action")
        # Use rsync to update the local file with the remote changes
        diff = context.client.sync.get_diff(self.path, self.local_metadata.signature)

        abs_path = context.workspace.datasites / self.path
        local_data = abs_path.read_bytes()
        new_data = py_fast_rsync.apply(local_data, diff.diff_bytes)
        new_hash = hashlib.sha256(new_data).hexdigest()

        if new_hash != diff.hash:
            # TODO error handling
            raise ValueError("Hash mismatch after applying diff")

        # TODO implement safe write with tempfile + rename
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(new_data)
        self.status = SyncStatus.SYNCED

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        # Client doesnt have read permission, so we do not apply any diff
        # This only happens in rare race-conditions where the permission is revoked after the action is determined.
        self.status = SyncStatus.REJECTED
        self.message = reason


class DeleteLocalAction(SyncAction):
    action_type = SyncActionType.DELETE_LOCAL

    def execute(self, context: SyftBoxContextInterface) -> None:
        abs_path = context.workspace.datasites / self.path
        abs_path.unlink()
        self.status = SyncStatus.SYNCED

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        # local delete cannot be rejected by server, this is a noop
        pass


class CreateRemoteAction(SyncAction):
    action_type = SyncActionType.CREATE_REMOTE

    def execute(self, context: SyftBoxContextInterface) -> None:
        abs_path = context.workspace.datasites / self.path
        data = abs_path.read_bytes()
        context.client.sync.create(self.path, data)
        self.status = SyncStatus.SYNCED

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        # Attempted upload without permission, the local file is renamed to a rejected file
        abs_path = context.workspace.datasites / self.path
        rejected_abs_path = format_rejected_path(abs_path)
        shutil.move(abs_path, rejected_abs_path)
        self.status = SyncStatus.REJECTED
        self.message = reason


class ModifyRemoteAction(SyncAction):
    action_type = SyncActionType.MODIFY_REMOTE

    def execute(self, context: SyftBoxContextInterface) -> None:
        abs_path = context.workspace.datasites / self.path
        local_data = abs_path.read_bytes()
        if self.remote_metadata is None:
            raise ValueError("Remote metadata is required for modify remote action")
        diff = py_fast_rsync.diff(self.remote_metadata.signature_bytes, local_data)
        if self.local_metadata is None:
            raise ValueError("Local metadata is required for modify remote action")
        context.client.sync.apply_diff(
            relative_path=self.path,
            diff=diff,
            expected_hash=self.local_metadata.hash,
        )
        self.status = SyncStatus.SYNCED

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        # Client doesnt have write permission, so the local changes are rejected and reverted to the remote state
        abs_path = context.workspace.datasites / self.path
        rejected_abs_path = format_rejected_path(abs_path)
        shutil.move(abs_path, rejected_abs_path)

        create_local_action = CreateLocalAction(local_metadata=None, remote_metadata=self.remote_metadata)
        try:
            create_local_action.execute(context)
        except SyftPermissionError:
            # Could not download the remote file due to lack of permission,
            # so only the .rejected file is left locally
            create_local_action.process_rejection(context)
        self.status = SyncStatus.REJECTED
        self.message = reason


class DeleteRemoteAction(SyncAction):
    action_type = SyncActionType.DELETE_REMOTE

    def execute(self, context: SyftBoxContextInterface) -> None:
        context.client.sync.delete(self.path)
        self.status = SyncStatus.SYNCED

    def process_rejection(self, context: SyftBoxContextInterface, reason: Optional[str] = None) -> None:
        # User does not have permission to delete the remote file, the delete is reverted
        create_local_action = CreateLocalAction(local_metadata=None, remote_metadata=self.remote_metadata)
        try:
            create_local_action.execute(context)
        except SyftPermissionError:
            # Could not re-download the file due to lack of permissions,
            create_local_action.process_rejection(context)
        self.status = SyncStatus.REJECTED
        self.message = reason


def _validate_local_action(context: SyftBoxContextInterface, action: SyncAction) -> None:
    if action.action_type in {SyncActionType.DELETE_LOCAL, SyncActionType.NOOP}:
        return

    abs_path = context.workspace.datasites / action.path

    # Create/modify local without remote metadata is invalid
    if (
        action.action_type in {SyncActionType.CREATE_LOCAL, SyncActionType.MODIFY_LOCAL}
        and action.remote_metadata is None
    ):
        raise SyncValidationError(f"Attempted to sync file {abs_path} to local, but remote file data is missing.")

    # Create/modify local over max file size is invalid
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if (
        action.action_type in {SyncActionType.CREATE_LOCAL, SyncActionType.MODIFY_LOCAL}
        and action.remote_metadata is not None
        and action.remote_metadata.file_size > max_size_bytes
    ):
        raise SyncValidationError(f"File {abs_path} is larger than {MAX_FILE_SIZE_MB}MB.")


def _validate_remote_action(context: SyftBoxContextInterface, action: SyncAction) -> None:
    # No validation needed for delete or noop actions
    if action.action_type in {SyncActionType.DELETE_REMOTE, SyncActionType.NOOP}:
        return

    abs_path = context.workspace.datasites / action.path

    # Create/modify remote without local metadata is invalid
    if (
        action.action_type in {SyncActionType.CREATE_REMOTE, SyncActionType.MODIFY_REMOTE}
        and action.local_metadata is None
    ):
        raise SyncValidationError(f"Attempted to sync file {abs_path} to remote, but local file data is missing.")

    # Create/modify remote over max file size is invalid
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if (
        action.action_type in {SyncActionType.CREATE_REMOTE, SyncActionType.MODIFY_REMOTE}
        and action.local_metadata is not None
        and action.local_metadata.file_size > max_size_bytes
    ):
        raise SyncValidationError(f"File {abs_path} is larger than {MAX_FILE_SIZE_MB}MB.")

    # Create/modify with broken permissions is invalid
    if (
        action.action_type in {SyncActionType.CREATE_REMOTE, SyncActionType.MODIFY_REMOTE}
        and SyftPermission.is_permission_file(abs_path)
        and not SyftPermission.is_valid(abs_path, abs_path.parent)  # Path does not matter for validation
    ):
        raise SyncValidationError(f"Encountered invalid permission file {abs_path}.")


def validate_sync_action(context: SyftBoxContextInterface, action: SyncAction) -> None:
    """
    Validate if the action can be executed.

    Args:
        action (SyncAction): The action to validate.

    Raises:
        SyncValidationError: If the action is invalid.
    """
    if action.side_to_update == SyncSide.LOCAL:
        _validate_local_action(context, action)
    else:
        _validate_remote_action(context, action)
