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

    @raise_unavailable("Could not save the checkpoint to Drive.")
    def put_checkpoint(self, peer: str, data: bytes) -> str:
        return self._put(peer, CHECKPOINT_FILE_NAME, data)

    @raise_unavailable("Could not read the checkpoint from Drive.")
    def latest_checkpoint(self, peer: str) -> bytes | None:
        return self._get(peer, CHECKPOINT_FILE_NAME)

    @raise_unavailable("Could not save the rolling state to Drive.")
    def put_rolling_state(self, peer: str, data: bytes) -> str:
        return self._put(peer, ROLLING_STATE_FILE_NAME, data)

    @raise_unavailable("Could not read the rolling state from Drive.")
    def get_rolling_state(self, peer: str) -> bytes | None:
        return self._get(peer, ROLLING_STATE_FILE_NAME)

    @raise_unavailable("Could not remove the watcher state from Drive.")
    def reset(self, peer: str) -> None:
        peer_folder_id = self._peer_folder_id(peer)
        if peer_folder_id is not None:
            self._cached_client.client.delete_file_by_id(peer_folder_id)

    @raise_unavailable("Could not list the watcher state on Drive.")
    def peers_with_saved_state(self) -> list[str]:
        folder_id = self._folder_id()
        if folder_id is None:
            return []
        peer_folders = self._cached_client.client.find_folders(parent_id=folder_id)
        return [name for _, name in peer_folders]

    def _put(self, peer: str, name: str, data: bytes) -> str:
        client = self._cached_client.client
        folder_id = self._folder_id() or self._create_folder()
        peer_folder_id = self._find_one(peer, folder_id) or client.create_folder(
            peer, folder_id
        )
        payload, _ = client.create_file_payload(data)
        file_id = self._find_one(name, peer_folder_id)
        if file_id is None:
            body = {"name": name, "parents": [peer_folder_id]}
            return client.create_syft_file(body, payload)["id"]
        execute_with_retries(
            client.drive_service.files().update(fileId=file_id, media_body=payload)
        )
        return file_id

    def _get(self, peer: str, name: str) -> bytes | None:
        peer_folder_id = self._peer_folder_id(peer)
        if peer_folder_id is None:
            return None
        file_id = self._find_one(name, peer_folder_id)
        if file_id is None:
            return None
        return self._cached_client.client.download_file(file_id)

    def _peer_folder_id(self, peer: str) -> str | None:
        folder_id = self._folder_id()
        if folder_id is None:
            return None
        return self._find_one(peer, folder_id)

    def _find_one(self, name: str, parent_id: str) -> str | None:
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

    def _folder_name(self) -> str:
        return (
            f"{self._cached_client.email}-{SYFT_VERSION}{WATCHER_STATE_FOLDER_SUFFIX}"
        )

    def _folder_id(self) -> str | None:
        folders = self._cached_client.client.find_folders(
            name_contains=[
                f"{self._cached_client.email}-",
                WATCHER_STATE_FOLDER_SUFFIX,
            ],
            parent_id=self._cached_client.get_syftbox_folder_id(),
        )
        return self._cached_client.find_or_adopt_versioned_folder(
            folders, current_name=self._folder_name()
        )

    def _create_folder(self) -> str:
        return self._cached_client.client.create_folder(
            self._folder_name(), self._cached_client.get_syftbox_folder_id()
        )
