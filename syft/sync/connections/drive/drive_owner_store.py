"""The DO's own storage on Google Drive.

The event log is the personal folder /SyftBox/{version}#{email}. Checkpoints
and the rolling state have folders of their own under /SyftBox.
"""

from syft.sync.checkpoints.checkpoint import (
    CHECKPOINT_FILENAME_PREFIX,
    INCREMENTAL_CHECKPOINT_PREFIX,
    Checkpoint,
    IncrementalCheckpoint,
)
from syft.sync.checkpoints.rolling_state import (
    ROLLING_STATE_FILENAME_PREFIX,
    RollingState,
)
from syft.sync.connections.drive.gdrive_client import (
    SYFT_FILE_PROPERTY_KEY,
    SYFT_FILE_PROPERTY_VALUE,
)
from syft.sync.connections.drive.drive_cached_client import (
    SYFTBOX_FOLDER,
    DriveCachedClient,
    valid_events_from_file_metadatas,
)
from syft.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
    raise_unavailable,
)
from syft.sync.connections.drive.gdrive_utils import (
    gather_all_file_and_folder_ids_recursive,
)
from syft.sync.events.file_change_event import FileChangeEventsMessageFileName
from syft.sync.transport.protocols import OwnerStore
from syft.version import SYFT_VERSION


def _filter_valid_file_metadatas(
    file_metadatas: list[dict],
) -> list[dict]:
    res = []
    for file_metadata in file_metadatas:
        fname = file_metadata["name"]
        try:
            FileChangeEventsMessageFileName.from_string(fname)
        except ValueError:
            # The folder holds other files, so a name that is not an event
            # name is normal here. This method filters them out.
            continue
        res.append(file_metadata)
    return res


class DriveOwnerStore(OwnerStore):
    """The DO's own folders in /SyftBox: event log, checkpoints, rolling state."""

    def __init__(self, cached_client: DriveCachedClient):
        self._cached_client = cached_client
        # Rolling state caches for single-API-call optimization
        self._rolling_state_folder_id: str | None = None
        self._rolling_state_file_id: str | None = None
        # fname -> gdrive id
        self.personal_syftbox_event_id_cache: dict[str, str] = {}

    def clone_for_thread(self) -> "DriveOwnerStore":
        """A store over a new Drive client, for use from another thread, with
        this store's caches."""
        clone = DriveOwnerStore(self._cached_client.clone_for_thread())
        clone._rolling_state_folder_id = self._rolling_state_folder_id
        clone._rolling_state_file_id = self._rolling_state_file_id
        clone.personal_syftbox_event_id_cache = dict(
            self.personal_syftbox_event_id_cache
        )
        return clone

    def reset_caches(self) -> None:
        self._cached_client.reset_caches()
        self._rolling_state_folder_id = None
        self._rolling_state_file_id = None
        self.personal_syftbox_event_id_cache.clear()

    def append_event(self, name: str, data: bytes) -> str:
        """Write raw bytes to /SyftBox/myemail."""
        personal_syftbox_folder_id = (
            self._cached_client.get_personal_syftbox_folder_id()
        )
        file_metadata = {
            "name": name,
            "parents": [personal_syftbox_folder_id],
        }
        file_payload, _ = self._cached_client.client.create_file_payload(data)

        res = self._cached_client.client.create_syft_file(file_metadata, file_payload)
        gdrive_id = res.get("id")
        self.personal_syftbox_event_id_cache[name] = gdrive_id
        return gdrive_id

    def fetch(self, handle: str) -> bytes:
        """Download raw bytes by file ID."""
        return self._cached_client.client.download_file(handle)

    def list_event_handles(self, since_timestamp: float | None = None) -> list[str]:
        personal_syftbox_folder_id = (
            self._cached_client.get_personal_syftbox_folder_id()
        )
        file_metadatas = self._cached_client.get_file_metadatas_from_folder(
            personal_syftbox_folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = _filter_valid_file_metadatas(file_metadatas)
        oldest_first = sorted(
            valid_fname_objs,
            key=lambda f: (
                FileChangeEventsMessageFileName.from_string(f["name"]).timestamp
            ),
        )
        return [f["id"] for f in oldest_first]

    @raise_unavailable("Could not read the event log from Drive.")
    def fetch_all_events(self) -> list[bytes]:
        """Download all event files from /SyftBox/myemail as raw bytes."""
        personal_syftbox_folder_id = (
            self._cached_client.get_personal_syftbox_folder_id()
        )
        file_metadatas = self._cached_client.get_file_metadatas_from_folder(
            personal_syftbox_folder_id
        )
        valid_fname_objs = valid_events_from_file_metadatas(file_metadatas)

        result = []
        for fname_obj in sorted(valid_fname_objs, key=lambda x: x.timestamp):
            gdrive_id = [
                f for f in file_metadatas if f["name"] == fname_obj.as_string()
            ][0]["id"]
            result.append(self._cached_client.client.download_file(gdrive_id))
        return result

    @raise_unavailable("Could not read the event log from Drive.")
    def fetch_events_since(self, since_timestamp: float | None) -> list[bytes]:
        """Download event files created after a timestamp as raw bytes."""
        if since_timestamp is None:
            return self.fetch_all_events()
        personal_folder_id = self._cached_client.get_personal_syftbox_folder_id()
        file_metadatas = self._cached_client.get_file_metadatas_from_folder(
            personal_folder_id, since_timestamp=since_timestamp
        )

        valid_fname_objs = valid_events_from_file_metadatas(file_metadatas)

        result = []
        for fname_obj in sorted(valid_fname_objs, key=lambda x: x.timestamp):
            if fname_obj.timestamp <= since_timestamp:
                continue

            gdrive_id = [
                f for f in file_metadatas if f["name"] == fname_obj.as_string()
            ][0]["id"]

            result.append(self._cached_client.client.download_file(gdrive_id))

        return result

    def _get_checkpoints_folder_name(self) -> str:
        """Get the checkpoints folder name: {email}-{version}-checkpoints"""
        return f"{self._cached_client.email}-{SYFT_VERSION}-checkpoints"

    def _get_checkpoints_folder_id(self) -> str | None:
        """Find the checkpoints folder ID from Google Drive."""
        folders = self._cached_client.client.find_folders(
            name_contains=[f"{self._cached_client.email}-", "-checkpoints"],
            parent_id=self._cached_client.get_syftbox_folder_id(),
        )
        return self._cached_client.find_or_adopt_versioned_folder(
            folders, current_name=self._get_checkpoints_folder_name()
        )

    def _get_or_create_checkpoints_folder_id(self) -> str:
        """Get or create the checkpoints folder."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is not None:
            return folder_id
        # Create the folder
        folder_name = self._get_checkpoints_folder_name()
        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()
        return self._cached_client.client.create_folder(folder_name, syftbox_folder_id)

    def put_checkpoint(self, name: str, data: bytes) -> str:
        """Upload raw checkpoint bytes to Google Drive.

        Uploads the new checkpoint first, then deletes old ones.
        """
        folder_id = self._get_or_create_checkpoints_folder_id()
        payload, _ = self._cached_client.client.create_file_payload(data)

        file_metadata = {
            "name": name,
            "parents": [folder_id],
        }

        result = self._cached_client.client.create_syft_file(file_metadata, payload)

        # Only delete old checkpoints after successful upload
        self._delete_all_checkpoints(exclude_file_id=result.get("id"))

        return result.get("id")

    @raise_unavailable("Could not read the latest checkpoint from Drive.")
    def latest_checkpoint(self) -> bytes | None:
        """Download the latest checkpoint as raw bytes, or None."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return None

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{CHECKPOINT_FILENAME_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id, name)"
            )
        )
        items = results.get("files", [])

        if not items:
            return None

        latest_file = None
        latest_timestamp = -1.0
        for item in items:
            timestamp = Checkpoint.filename_to_timestamp(item["name"])
            if timestamp is not None and timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_file = item

        if latest_file is None:
            return None

        return self._cached_client.client.download_file(latest_file["id"])

    def _delete_all_checkpoints(self, exclude_file_id: str | None = None):
        """Delete all existing full checkpoints (not incremental ones).

        Args:
            exclude_file_id: If provided, skip deleting this file ID
                (used to preserve a newly uploaded checkpoint).
        """
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return

        # List only full checkpoint files (start with "checkpoint_" not "incremental_checkpoint_")
        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{CHECKPOINT_FILENAME_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id, name)"
            )
        )
        items = results.get("files", [])

        # Delete only full checkpoints (not incremental ones)
        for item in items:
            if item["name"].startswith(INCREMENTAL_CHECKPOINT_PREFIX):
                continue  # Skip incremental checkpoints
            if item["id"] == exclude_file_id:
                continue  # Skip the newly uploaded checkpoint
            try:
                execute_with_retries(
                    self._cached_client.client.drive_service.files().delete(
                        fileId=item["id"]
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to delete checkpoint {item['name']}: {e}")

    def put_incremental(self, name: str, data: bytes) -> str:
        """Upload raw incremental checkpoint bytes to Google Drive."""
        folder_id = self._get_or_create_checkpoints_folder_id()
        payload, _ = self._cached_client.client.create_file_payload(data)

        file_metadata = {
            "name": name,
            "parents": [folder_id],
        }

        result = self._cached_client.client.create_syft_file(file_metadata, payload)
        return result.get("id")

    @raise_unavailable("Could not read the incremental checkpoints from Drive.")
    def all_incrementals(self) -> list[bytes]:
        """Download all incremental checkpoints as raw bytes."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return []

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id, name)"
            )
        )
        items = results.get("files", [])

        if not items:
            return []

        return [self._cached_client.client.download_file(item["id"]) for item in items]

    def incremental_count(self) -> int:
        """Get the number of incremental checkpoints on Google Drive."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return 0

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id)"
            )
        )
        return len(results.get("files", []))

    def next_incremental_sequence(self) -> int:
        """Get the next sequence number for incremental checkpoints."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return 1

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(name)"
            )
        )
        items = results.get("files", [])

        if not items:
            return 1

        # Find highest sequence number
        max_seq = 0
        for item in items:
            seq = IncrementalCheckpoint.filename_to_sequence_number(item["name"])
            if seq is not None and seq > max_seq:
                max_seq = seq

        return max_seq + 1

    def delete_incrementals(self) -> None:
        """Delete all incremental checkpoints (called after compacting)."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id, name)"
            )
        )
        items = results.get("files", [])

        for item in items:
            try:
                execute_with_retries(
                    self._cached_client.client.drive_service.files().delete(
                        fileId=item["id"]
                    )
                )
            except Exception as e:
                print(
                    f"Warning: Failed to delete incremental checkpoint {item['name']}: {e}"
                )

    def _get_rolling_state_folder_name(self) -> str:
        """Get the rolling state folder name: {email}-{version}-rolling-state"""
        return f"{self._cached_client.email}-{SYFT_VERSION}-rolling-state"

    def _get_rolling_state_folder_id(self, use_cache: bool = True) -> str | None:
        """
        Find the rolling state folder ID from Google Drive.

        Args:
            use_cache: If True, return cached value if available.
        """
        if use_cache and self._rolling_state_folder_id is not None:
            return self._rolling_state_folder_id

        folders = self._cached_client.client.find_folders(
            name_contains=[f"{self._cached_client.email}-", "-rolling-state"],
            parent_id=self._cached_client.get_syftbox_folder_id(),
        )
        folder_id = self._cached_client.find_or_adopt_versioned_folder(
            folders, current_name=self._get_rolling_state_folder_name()
        )
        if folder_id is not None:
            self._rolling_state_folder_id = folder_id
        return folder_id

    def _get_or_create_rolling_state_folder_id(self) -> str:
        """Get or create the rolling state folder."""
        folder_id = self._get_rolling_state_folder_id()
        if folder_id is not None:
            return folder_id
        # Create the folder
        folder_name = self._get_rolling_state_folder_name()
        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()
        folder_id = self._cached_client.client.create_folder(
            folder_name, syftbox_folder_id
        )
        self._rolling_state_folder_id = folder_id
        return folder_id

    def put_rolling_state(self, name: str, data: bytes) -> str:
        """Upload raw rolling state bytes to Google Drive.

        Optimized to use update() if file ID is cached.
        """
        payload, _ = self._cached_client.client.create_file_payload(data)

        # Try to update existing file if we have a cached ID
        if self._rolling_state_file_id is not None:
            try:
                execute_with_retries(
                    self._cached_client.client.drive_service.files().update(
                        fileId=self._rolling_state_file_id,
                        media_body=payload,
                    )
                )
                return self._rolling_state_file_id
            except Exception as e:
                # The cached file is gone or unreachable. Clear the cache and
                # write a new file below.
                print(
                    f"Warning: could not update rolling state "
                    f"{self._rolling_state_file_id}, writing a new file: {e}"
                )
                self._rolling_state_file_id = None

        folder_id = self._get_or_create_rolling_state_folder_id()

        file_metadata = {
            "name": name,
            "parents": [folder_id],
        }

        result = self._cached_client.client.create_syft_file(file_metadata, payload)
        file_id = result.get("id")
        self._rolling_state_file_id = file_id
        return file_id

    @raise_unavailable("Could not read the rolling state from Drive.")
    def get_rolling_state(self) -> bytes | None:
        """Download the latest rolling state as raw bytes, or None.

        Also populates the folder and file ID caches for subsequent uploads.
        """
        folder_id = self._get_rolling_state_folder_id()
        if folder_id is None:
            return None

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{ROLLING_STATE_FILENAME_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id, name)"
            )
        )
        items = results.get("files", [])

        if not items:
            return None

        latest_file = None
        latest_timestamp = -1.0
        for item in items:
            timestamp = RollingState.filename_to_timestamp(item["name"])
            if timestamp is not None and timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_file = item

        if latest_file is None:
            return None

        data = self._cached_client.client.download_file(latest_file["id"])
        self._rolling_state_file_id = latest_file["id"]
        return data

    def delete_rolling_state(self) -> None:
        """Delete all existing rolling state files and clear cache."""
        # Clear the file ID cache
        self._rolling_state_file_id = None

        folder_id = self._get_rolling_state_folder_id()
        if folder_id is None:
            return

        # List all rolling state files
        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{ROLLING_STATE_FILENAME_PREFIX}'"
        )
        results = execute_with_retries(
            self._cached_client.client.drive_service.files().list(
                q=query, fields="files(id, name)"
            )
        )
        items = results.get("files", [])

        # Delete each rolling state file
        for item in items:
            try:
                execute_with_retries(
                    self._cached_client.client.drive_service.files().delete(
                        fileId=item["id"]
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to delete rolling state {item['name']}: {e}")

    # -- the whole account -----------------------------------------------------

    def delete_all(self) -> int:
        """Everything in /SyftBox, and the syft files Drive moved to the root
        of My Drive when they lost their folder."""
        file_ids = set(self.gather_all_file_and_folder_ids())
        file_ids |= set(self.find_orphaned_syft_files())
        self._cached_client.client.delete_multiple_files_by_ids(list(file_ids))
        return len(file_ids)

    def gather_all_file_and_folder_ids(self) -> list[str]:
        """Every file and folder under /SyftBox, and /SyftBox itself."""
        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()
        return gather_all_file_and_folder_ids_recursive(
            self._cached_client.client.drive_service, syftbox_folder_id
        )

    def find_orphaned_syft_files(self) -> list[str]:
        """Find syft files of this account in its My Drive root, except /SyftBox.

        Drive moves a file to the root of its owner when the file loses its
        folder. Only /SyftBox belongs in root.

        Returns list of orphaned file IDs.
        """
        query = (
            "'root' in parents and 'me' in owners and trashed=false and "
            f"name != '{SYFTBOX_FOLDER}' and "
            f"properties has {{ key='{SYFT_FILE_PROPERTY_KEY}' "
            f"and value='{SYFT_FILE_PROPERTY_VALUE}' }}"
        )
        return [
            f["id"]
            for f in self._cached_client.client.list_all_files(query, fields="id")
        ]
