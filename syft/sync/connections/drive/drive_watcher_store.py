"""The DS's own storage on Google Drive.

The folder /SyftBox/{email}-{version}-watcher-state is shared with nobody. It
holds one folder per DO, and each DO folder holds one checkpoint file and one
rolling-state file. A save replaces the content of the file in place, so a
read gets the old content or the new content, never a mix.
"""

from syft.sync.connections.drive.drive_cached_client import DriveCachedClient
from syft.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
    raise_unavailable,
)
from syft.sync.connections.protocols import WatcherStoreConnection
from syft.version import SYFT_VERSION

WATCHER_STATE_FOLDER_SUFFIX = "-watcher-state"
CHECKPOINT_FILE_NAME = "checkpoint.tar.gz"
ROLLING_STATE_FILE_NAME = "rolling_state.tar.gz"


class DriveWatcherStoreConnection(WatcherStoreConnection):
    """The DS's folder in /SyftBox: one checkpoint and one rolling state per DO."""

    def __init__(self, cached_client: DriveCachedClient):
        self._cached_client = cached_client

    def clone_for_thread(self) -> "DriveWatcherStoreConnection":
        """A connection over a new Drive client, for use from another thread."""
        return DriveWatcherStoreConnection(self._cached_client.clone_for_thread())

    @raise_unavailable("Could not save the checkpoint to Drive.")
    def put_checkpoint(self, peer: str, data: bytes) -> None:
        self._save_file(peer, CHECKPOINT_FILE_NAME, data)

    @raise_unavailable("Could not read the checkpoint from Drive.")
    def get_checkpoint(self, peer: str) -> bytes | None:
        return self._read_file(peer, CHECKPOINT_FILE_NAME)

    @raise_unavailable("Could not save the rolling state to Drive.")
    def put_rolling_state(self, peer: str, data: bytes) -> None:
        self._save_file(peer, ROLLING_STATE_FILE_NAME, data)

    @raise_unavailable("Could not read the rolling state from Drive.")
    def get_rolling_state(self, peer: str) -> bytes | None:
        return self._read_file(peer, ROLLING_STATE_FILE_NAME)

    @raise_unavailable("Could not remove the watcher state from Drive.")
    def reset(self, peer: str) -> None:
        peer_folder_id = self._peer_folder_id(peer)
        if peer_folder_id is not None:
            self._cached_client.client.delete_file_by_id(peer_folder_id)

    @raise_unavailable("Could not list the watcher state on Drive.")
    def peers_with_saved_state(self) -> list[str]:
        watcher_folder_id = self._watcher_folder_id()
        if watcher_folder_id is None:
            return []
        peer_folders = self._cached_client.client.find_folders(
            parent_id=watcher_folder_id
        )
        return [name for _, name in peer_folders]

    def _save_file(self, peer: str, name: str, data: bytes) -> None:
        client = self._cached_client.client
        watcher_folder_id = self._watcher_folder_id() or self._create_watcher_folder()
        peer_folder_id = self._find_id(peer, watcher_folder_id) or client.create_folder(
            peer, watcher_folder_id
        )
        payload, _ = client.create_file_payload(data)
        file_id = self._find_id(name, peer_folder_id)
        if file_id is None:
            body = {"name": name, "parents": [peer_folder_id]}
            client.create_syft_file(body, payload)
        else:
            execute_with_retries(
                client.drive_service.files().update(fileId=file_id, media_body=payload)
            )

    def _read_file(self, peer: str, name: str) -> bytes | None:
        peer_folder_id = self._peer_folder_id(peer)
        if peer_folder_id is None:
            return None
        file_id = self._find_id(name, peer_folder_id)
        if file_id is None:
            return None
        return self._cached_client.client.download_file(file_id)

    def _peer_folder_id(self, peer: str) -> str | None:
        watcher_folder_id = self._watcher_folder_id()
        if watcher_folder_id is None:
            return None
        return self._find_id(peer, watcher_folder_id)

    def _find_id(self, name: str, parent_id: str) -> str | None:
        """The id of the file or folder ``name`` in ``parent_id``, or None.

        Raises:
            RuntimeError: if the folder holds more than one of that name.
        """
        query = f"name='{name}' and '{parent_id}' in parents and trashed=false"
        files = self._cached_client.client.list_all_files(query, fields="id")
        if len(files) > 1:
            raise RuntimeError(
                f"Found {len(files)} files named {name} in one folder of the "
                "watcher state on Drive. Exactly one is expected."
            )
        return files[0]["id"] if files else None

    def _watcher_folder_name(self) -> str:
        return (
            f"{self._cached_client.email}-{SYFT_VERSION}{WATCHER_STATE_FOLDER_SUFFIX}"
        )

    def _watcher_folder_id(self) -> str | None:
        folders = self._cached_client.client.find_folders(
            name_contains=[
                f"{self._cached_client.email}-",
                WATCHER_STATE_FOLDER_SUFFIX,
            ],
            parent_id=self._cached_client.get_syftbox_folder_id(),
        )
        return self._cached_client.find_or_adopt_versioned_folder(
            folders, current_name=self._watcher_folder_name()
        )

    def _create_watcher_folder(self) -> str:
        return self._cached_client.client.create_folder(
            self._watcher_folder_name(), self._cached_client.get_syftbox_folder_id()
        )
