"""Google Drive Files transport layer implementation"""

import logging
import io
import json
from pathlib import Path
import pickle
from syft_client.sync.utils.syftbox_utils import check_env
from syft_client.version import SYFT_CLIENT_VERSION
from typing import Any, Dict, List, Optional, Tuple
from typing import TYPE_CHECKING
from pydantic import BaseModel
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload, build_http
from google.oauth2.credentials import Credentials as GoogleCredentials

from syft_client.sync.connections.drive.gdrive_utils import (
    gather_all_file_and_folder_ids_recursive,
)
from syft_client.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
    next_chunk_with_retries,
    batch_execute_with_retries,
)
from syft_client.sync.version.version_info import _parse_semver

from syft_client.sync.connections.base_connection import (
    FileCollection,
    SyftboxPlatformConnection,
)
from syft_datasets.dataset_manager import (
    DATASET_COLLECTION_PREFIX,
    PRIVATE_DATASET_COLLECTION_PREFIX,
)
from syft_client.sync.events.file_change_event import (
    FileChangeEventsMessageFileName,
    FileChangeEventsMessage,
)
from syft_client.sync.messages.proposed_filechange import (
    MessageFileName,
    FileNameParseError,
    ProposedFileChangesMessage,
)
from syft_client.sync.environments.environment import Environment
from syft_client.sync.checkpoints.checkpoint import (
    Checkpoint,
    IncrementalCheckpoint,
    CHECKPOINT_FILENAME_PREFIX,
    INCREMENTAL_CHECKPOINT_PREFIX,
)
from syft_client.sync.checkpoints.rolling_state import (
    RollingState,
    ROLLING_STATE_FILENAME_PREFIX,
)

if TYPE_CHECKING:
    from syft_client.sync.connections.drive.grdrive_config import (
        GdriveConnectionConfig,
    )
    from syft_client.sync.version.version_info import VersionInfo

# Timeout for Google API requests (in seconds)
GOOGLE_API_TIMEOUT = 120  # 2 minutes

SYFTBOX_FOLDER = "SyftBox"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]
logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)


def build_drive_service(
    credentials: GoogleCredentials,
    timeout: int = GOOGLE_API_TIMEOUT,
    environment: Environment | None = None,
):
    """Build a Google Drive service with timeout-enabled authorized HTTP."""
    # Build the http via googleapiclient's own factory rather than constructing
    # httplib2.Http() directly. build_http() applies Google-API-specific tweaks.
    http = build_http()
    http.timeout = timeout
    if environment == Environment.COLAB:
        from google.colab import auth as colab_auth
        import google.auth

        colab_auth.authenticate_user()
        creds, _ = google.auth.default()
        authed_http = AuthorizedHttp(creds, http=http)
        # Build service without explicit credentials in Colab
        return build("drive", "v3", http=authed_http)
    else:
        authorized_http = AuthorizedHttp(credentials, http=http)
        return build("drive", "v3", http=authorized_http)


LEGACY_GDRIVE_OUTBOX_INBOX_FOLDER_PREFIX = "syft_outbox_inbox"  # legacy prefix
GDRIVE_P2P_FOLDER_DATASITE_PREFIX = "syft_datasite"
SYFT_PEERS_FILE = "SYFT_peers.json"
SYFT_VERSION_FILE = "SYFT_version.json"


class GdriveArchiveFolder(BaseModel):
    sender_email: str
    recipient_email: str

    def as_string(self) -> str:
        return f"syft_{self.sender_email}_to_{self.recipient_email}_archive"


class GdriveP2PFolder(BaseModel):
    """Folder for peer communication: syft_datasite#version#datasite_email#inbox|outbox#peer_email"""

    datasite_email: str
    folder_type: str  # "inbox" or "outbox"
    peer_email: str

    def as_string(self) -> str:
        return f"{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#{SYFT_CLIENT_VERSION}#{self.datasite_email}#{self.folder_type}#{self.peer_email}"

    @classmethod
    def from_name(cls, name: str) -> "GdriveP2PFolder":
        parts = name.split("#")
        if len(parts) != 5 or parts[0] != GDRIVE_P2P_FOLDER_DATASITE_PREFIX:
            raise ValueError(f"Invalid P2P folder name: {name}")
        return cls(datasite_email=parts[2], folder_type=parts[3], peer_email=parts[4])


class GdriveEncryptionBundlesFolder(BaseModel):
    """Folder for encryption bundles: syft_encryption_bundles#email"""

    email: str

    def as_string(self) -> str:
        return f"syft_encryption_bundles#{self.email}"


class GdrivePersonalSyftboxFolder(BaseModel):
    """Folder for personal SyftBox: {version}#{email} under /SyftBox/"""

    email: str

    def as_string(self) -> str:
        return f"{SYFT_CLIENT_VERSION}#{self.email}"


class DatasetCollectionFolder(BaseModel):
    """Represents a dataset collection folder with format: {prefix}_{tag}_{hash}"""

    tag: str
    content_hash: str

    def as_string(self) -> str:
        return f"{DATASET_COLLECTION_PREFIX}_{self.tag}_{self.content_hash}"

    @classmethod
    def from_name(cls, name: str) -> "DatasetCollectionFolder":
        """Parse folder name like 'syft_datasetcollection_mytag_abc123'"""
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid dataset collection folder name: {name}")
        # prefix is parts[0:2] joined = "syft_datasetcollection"
        # tag is parts[2:-1] joined (in case tag has underscores)
        # hash is parts[-1]
        tag = "_".join(parts[2:-1])
        content_hash = parts[-1]
        return cls(tag=tag, content_hash=content_hash)

    @staticmethod
    def compute_hash(files: dict[str, bytes]) -> str:
        """Compute a hash from file contents."""
        from syft_client.sync.file_utils import compute_file_hashes

        return compute_file_hashes(files)


class PrivateDatasetCollectionFolder(BaseModel):
    """Represents a private dataset collection folder with format: {prefix}_{tag}_{hash}"""

    tag: str
    content_hash: str

    def as_string(self) -> str:
        return f"{PRIVATE_DATASET_COLLECTION_PREFIX}_{self.tag}_{self.content_hash}"

    @classmethod
    def from_name(cls, name: str) -> "PrivateDatasetCollectionFolder":
        """Parse folder name like 'syft_privatecollection_mytag_abc123'"""
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid private collection folder name: {name}")
        tag = "_".join(parts[2:-1])
        content_hash = parts[-1]
        return cls(tag=tag, content_hash=content_hash)

    @staticmethod
    def compute_hash(files: dict[str, bytes]) -> str:
        """Compute a hash from file contents."""
        from syft_client.sync.file_utils import compute_file_hashes

        return compute_file_hashes(files)


# Helpers for finding folders whose names embed SYFT_CLIENT_VERSION. Folder
# names use '#' or '-' as field separators with the version as one field,
# so we walk those fields looking for an X.Y.Z-shaped chunk -- no per-format
# parser needed. Patch-compat means same major.minor (matches the protocol
# definition in version_info.is_compatible_with).


def _looks_like_version(s: str) -> bool:
    """True if s is shaped like 'X.Y.Z' with all numeric parts."""
    parts = s.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def _extract_version_from_name(name: str) -> str | None:
    """Return the first '#'/'-'-separated field that looks like a semver."""
    for chunk in name.replace("-", "#").split("#"):
        if _looks_like_version(chunk):
            return chunk
    return None


def _filter_patch_compatible(
    folders: list[tuple[str, str]],
    current_version: str | None = None,
) -> list[tuple[str, str]]:
    """Keep folders whose embedded version has matching major.minor.

    `current_version` defaults to the module-level SYFT_CLIENT_VERSION at call
    time (not import time) so tests that patch the version take effect.
    """
    if current_version is None:
        current_version = SYFT_CLIENT_VERSION
    try:
        cur_major, cur_minor, _ = _parse_semver(current_version)
    except ValueError:
        return []
    kept: list[tuple[str, str]] = []
    for fid, name in folders:
        version_str = _extract_version_from_name(name)
        if version_str is None:
            continue
        try:
            major, minor, _ = _parse_semver(version_str)
        except ValueError:
            continue
        if major == cur_major and minor == cur_minor:
            kept.append((fid, name))
    return kept


class GDriveConnection(SyftboxPlatformConnection):
    """Google Drive Files API transport layer"""

    class Config:
        arbitrary_types_allowed = True

    drive_service: Any = None
    credentials: GoogleCredentials | None = None
    verbose: bool = True
    email: str
    token_path: Path | None = None
    _is_setup: bool = False

    # /SyftBox
    # this is the toplevel folder with inboxes, outboxes and personal syftbox
    _syftbox_folder_id: str | None = None

    # /SyftBox/myemail
    # this is where we store the personal data
    _personal_syftbox_folder_id: str | None = None

    # peer_email -> folder_id (folders I created for peer's datasite)
    peer_datasite_inbox_cache: Dict[str, str] = {}
    peer_datasite_outbox_cache: Dict[str, str] = {}

    # peer_email -> folder_id (folders peer created for my datasite)
    own_datasite_inbox_cache: Dict[str, str] = {}
    own_datasite_outbox_cache: Dict[str, str] = {}

    # sender email -> archive folder id
    archive_folder_id_cache: Dict[str, str] = {}

    # fname -> gdrive id
    personal_syftbox_event_id_cache: Dict[str, str] = {}

    # tag -> dataset collection folder id
    dataset_collection_folder_id_cache: Dict[str, str] = {}

    # Rolling state caches for single-API-call optimization
    _rolling_state_folder_id: str | None = None
    _rolling_state_file_id: str | None = None

    # Encryption bundles folder cache
    _encryption_bundles_folder_id: str | None = None

    # Cached SYFT_peers.json contents (None = not loaded yet).
    _peers_json_cache: Dict[str, Dict[str, str]] | None = None

    @classmethod
    def from_config(cls, config: "GdriveConnectionConfig") -> "GDriveConnection":
        return cls.from_token_path(config.email, config.token_path)

    @classmethod
    def from_token_path(
        cls,
        email: str,
        token_path: Path | None,
        warm_syftbox_folder_id_cache: bool = True,
    ) -> "GDriveConnection":
        res = cls(email=email, token_path=token_path)
        if token_path:
            credentials = GoogleCredentials.from_authorized_user_file(
                token_path, SCOPES
            )
        else:
            credentials = None
        res.setup(
            credentials=credentials,
            warm_syftbox_folder_id_cache=warm_syftbox_folder_id_cache,
        )
        return res

    @classmethod
    def from_service(cls, email: str, mock_service: Any) -> "GDriveConnection":
        """Create a GDriveConnection using a mock drive service for testing.

        Args:
            email: Email of the user
            mock_service: MockDriveService instance to use instead of real API

        Returns:
            GDriveConnection configured with the mock service
        """
        from syft_client.sync.connections.drive.mock_drive_service import (
            MockDriveService,
        )

        res = cls(email=email, token_path=None)
        if isinstance(mock_service, MockDriveService):
            mock_service = MockDriveService(mock_service._backing_store, email)
        res.setup(drive_service=mock_service)
        return res

    def setup(
        self,
        credentials: GoogleCredentials | None = None,
        drive_service: Any | None = None,
        warm_syftbox_folder_id_cache: bool = True,
    ):
        """Setup Drive transport with OAuth2 credentials, Colab auth, or mock service.

        Args:
            credentials: OAuth2 credentials for real API access
            drive_service: Drive service instance (e.g., mock for testing)
            warm_syftbox_folder_id_cache: If True, eagerly resolve the personal
                SyftBox folder id (2-3 API calls). Pass False when caches are
                seeded by the caller (e.g. ``copy()``).
        """
        self.credentials = credentials
        if drive_service is not None:
            self.drive_service = drive_service
        else:
            self.drive_service = build_drive_service(
                self.credentials, environment=self.environment
            )

        if warm_syftbox_folder_id_cache:
            self.get_personal_syftbox_folder_id()
        self._is_setup = True

    def copy(self) -> "GDriveConnection":
        # if is mock
        from syft_client.sync.connections.drive.mock_drive_service import (
            MockDriveService,
        )

        if isinstance(self.drive_service, MockDriveService):
            new_conn = GDriveConnection.from_service(self.email, self.drive_service)
        else:
            new_conn = GDriveConnection.from_token_path(
                self.email, self.token_path, warm_syftbox_folder_id_cache=False
            )
        self._copy_caches_to(new_conn)
        return new_conn

    def _copy_caches_to(self, other: "GDriveConnection") -> None:
        """Seed all in-memory caches on `other` from `self`, so a freshly-built
        copy doesn't re-discover folder ids via Drive search on first use."""
        other._syftbox_folder_id = self._syftbox_folder_id
        other._personal_syftbox_folder_id = self._personal_syftbox_folder_id
        other._rolling_state_folder_id = self._rolling_state_folder_id
        other._rolling_state_file_id = self._rolling_state_file_id
        other._encryption_bundles_folder_id = self._encryption_bundles_folder_id
        other._peers_json_cache = (
            dict(self._peers_json_cache) if self._peers_json_cache is not None else None
        )
        other.peer_datasite_inbox_cache = dict(self.peer_datasite_inbox_cache)
        other.peer_datasite_outbox_cache = dict(self.peer_datasite_outbox_cache)
        other.own_datasite_inbox_cache = dict(self.own_datasite_inbox_cache)
        other.own_datasite_outbox_cache = dict(self.own_datasite_outbox_cache)
        other.archive_folder_id_cache = dict(self.archive_folder_id_cache)
        other.personal_syftbox_event_id_cache = dict(
            self.personal_syftbox_event_id_cache
        )
        other.dataset_collection_folder_id_cache = dict(
            self.dataset_collection_folder_id_cache
        )

    @property
    def environment(self) -> Environment:
        return check_env()

    def get_authenticated_email(self) -> str:
        """Return the email of the Google account behind drive_service."""
        about = execute_with_retries(
            self.drive_service.about().get(fields="user(emailAddress)")
        )
        return about["user"]["emailAddress"]

    def create_personal_syftbox_folder(self) -> str:
        """Creates /SyftBox/{version}#{email}"""
        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_name = GdrivePersonalSyftboxFolder(email=self.email).as_string()
        return self.create_folder(folder_name, syftbox_folder_id)

    def create_syftbox_folder(self) -> str:
        """Creates /SyftBox"""
        return self.create_folder(SYFTBOX_FOLDER, None)

    def create_archive_folder(self, sender_email: str) -> str:
        archive_folder = GdriveArchiveFolder(
            sender_email=sender_email, recipient_email=self.email
        )
        archive_folder_name = archive_folder.as_string()
        syftbox_folder_id = self.get_syftbox_folder_id()
        return self.create_folder(archive_folder_name, syftbox_folder_id)

    def create_peer_datasite_folders(self, peer_email: str):
        """Create inbox + outbox folders for peer's datasite on own drive, share with peer.

        Creates:
        - syft_datasite_{peer}_inbox_{self} — I submit proposals to peer here
        - syft_datasite_{peer}_outbox_{self} — peer pushes events to me here
        """
        # Inbox: I write proposals, peer reads
        inbox_id = self._get_peer_datasite_inbox_id(peer_email)
        if inbox_id is None:
            inbox_id = self._create_peer_datasite_folder(peer_email, "inbox")
        self.add_permission(inbox_id, peer_email, write=True)

        # Outbox: peer writes events, I read
        outbox_id = self._get_peer_datasite_outbox_id(peer_email)
        if outbox_id is None:
            outbox_id = self._create_peer_datasite_folder(peer_email, "outbox")
        self.add_permission(outbox_id, peer_email, write=True)

    def add_peer(self, peer_email: str):
        """Alias for create_peer_datasite_folders."""
        self.create_peer_datasite_folders(peer_email)

    def _get_peers_file_id(self) -> str | None:
        """Find SYFT_peers.json file in /SyftBox folder"""
        syftbox_folder_id = self.get_syftbox_folder_id()
        query = f"name='{SYFT_PEERS_FILE}' and '{syftbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def _download_peers_json(self) -> Dict[str, Dict[str, str]]:
        """Fetch peers JSON from GDrive. Returns empty dict if not found."""
        file_id = self._get_peers_file_id()
        if file_id is None:
            return {}

        try:
            file_data = self.download_file(file_id)
            return json.loads(file_data.decode("utf-8"))
        except Exception as e:
            print(f"Warning: Error reading peers file: {e}")
            return {}

    def _get_peers_json(
        self, force_download: bool = False
    ) -> Dict[str, Dict[str, str]]:
        """Return peers JSON, using the in-memory cache when available."""
        if self._peers_json_cache is not None and not force_download:
            return self._peers_json_cache
        self._peers_json_cache = self._download_peers_json()
        return self._peers_json_cache

    def _write_peers_json(self, peers_data: Dict[str, Dict[str, str]]):
        """Write peers JSON to GDrive. Creates or updates the file."""
        syftbox_folder_id = self.get_syftbox_folder_id()
        file_id = self._get_peers_file_id()

        # Convert to JSON bytes
        json_data = json.dumps(peers_data, indent=2)
        file_payload, _ = self.create_file_payload(json_data)

        if file_id is None:
            # Create new file
            file_metadata = {
                "name": SYFT_PEERS_FILE,
                "parents": [syftbox_folder_id],
            }
            result = execute_with_retries(
                self.drive_service.files().create(
                    body=file_metadata, media_body=file_payload, fields="id"
                )
            )
            self._peers_json_cache = peers_data
            return result.get("id")
        else:
            # Update existing file
            execute_with_retries(
                self.drive_service.files().update(
                    fileId=file_id, media_body=file_payload
                )
            )
            self._peers_json_cache = peers_data
            return file_id

    def _update_peer_state(
        self,
        peer_email: str,
        state: str,
        public_encryption_bundle: dict | None = None,
    ):
        """Update a single peer's state in the JSON file.

        Preserves existing fields when updating state.
        """
        peers_data = self._get_peers_json()
        existing = peers_data.get(peer_email, {})
        existing["state"] = state
        if public_encryption_bundle is not None:
            existing["public_encryption_bundle"] = public_encryption_bundle
        peers_data[peer_email] = existing
        self._write_peers_json(peers_data)

    def get_peer_requests(self) -> List[str]:
        """Get list of pending peer requests.

        Scans for syft_datasite_#version#{self}_*  folders NOT owned by self — those are
        peers who created folders for our datasite. Filters out already-accepted
        or rejected peers from SYFT_peers.json.
        """
        results = execute_with_retries(
            self.drive_service.files().list(
                q=f"name contains '{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#' "
                f"and name contains '#{self.email}#' "
                f"and trashed=false "
                f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
                f"and not 'me' in owners"
            )
        )

        all_folder_peers = set()
        for f in results.get("files", []):
            try:
                folder = GdriveP2PFolder.from_name(f["name"])
                if folder.datasite_email == self.email:
                    all_folder_peers.add(folder.peer_email)
            except (ValueError, Exception):
                continue

        peers_data = self._get_peers_json()
        pending_peers = []
        for peer_email in all_folder_peers:
            if peer_email not in peers_data:
                pending_peers.append(peer_email)
            elif peers_data[peer_email].get("state") not in ["accepted", "rejected"]:
                pending_peers.append(peer_email)

        return pending_peers

    def watcher_download_raw_events_from_outbox(
        self, peer_email: str, since_timestamp: float | None
    ) -> list[bytes]:
        folder_id = self._get_peer_datasite_outbox_id(peer_email)
        if folder_id is None:
            return []

        file_metadatas = self.get_file_metadatas_from_folder(
            folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = self._get_valid_events_from_file_metadatas(file_metadatas)
        name_to_id = {f["name"]: f["id"] for f in file_metadatas}

        sorted_fname_objs = [
            x
            for x in sorted(valid_fname_objs, key=lambda x: x.timestamp)
            if since_timestamp is None or x.timestamp > since_timestamp
        ]

        if len(sorted_fname_objs) == 0:
            return []

        res = []
        for fname_obj in sorted_fname_objs:
            file_name = fname_obj.as_string()
            if file_name in name_to_id:
                res.append(self.download_file(name_to_id[file_name]))
        return res

    def watcher_get_events_messages(
        self, peer_email: str, since_timestamp: float | None
    ) -> List[FileChangeEventsMessage]:
        raw_list = self.watcher_download_raw_events_from_outbox(
            peer_email, since_timestamp
        )
        return [FileChangeEventsMessage.from_compressed_data(data) for data in raw_list]

    def watcher_get_outbox_file_metadatas(
        self, peer_email: str, since_timestamp: float | None
    ) -> List[Dict]:
        """Get file metadata from peer's outbox folder without downloading."""
        folder_id = self._get_peer_datasite_outbox_id(peer_email)
        if folder_id is None:
            return []

        file_metadatas = self.get_file_metadatas_from_folder(
            folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = self._get_valid_events_from_file_metadatas(file_metadatas)
        name_to_id = {f["name"]: f["id"] for f in file_metadatas}

        result = []
        for fname_obj in sorted(valid_fname_objs, key=lambda x: x.timestamp):
            if since_timestamp is None or fname_obj.timestamp > since_timestamp:
                file_name = fname_obj.as_string()
                if file_name in name_to_id:
                    result.append(
                        {
                            "file_id": name_to_id[file_name],
                            "file_name": file_name,
                            "timestamp": fname_obj.timestamp,
                        }
                    )
        return result

    def owner_write_raw_bytes_to_syftbox(self, filename: str, data: bytes) -> str:
        """Write raw bytes to /SyftBox/myemail."""
        personal_syftbox_folder_id = self.get_personal_syftbox_folder_id()
        file_metadata = {
            "name": filename,
            "parents": [personal_syftbox_folder_id],
        }
        file_payload, _ = self.create_file_payload(data)

        res = execute_with_retries(
            self.drive_service.files().create(
                body=file_metadata, media_body=file_payload, fields="id"
            )
        )
        gdrive_id = res.get("id")
        self.personal_syftbox_event_id_cache[filename] = gdrive_id
        return gdrive_id

    def owner_download_raw_bytes_by_id(self, file_id: str) -> bytes:
        """Download raw bytes by file ID."""
        return self.download_file(file_id)

    def owner_get_all_accepted_event_file_ids(
        self, since_timestamp: float | None = None
    ) -> List[str]:
        personal_syftbox_folder_id = self.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(
            personal_syftbox_folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = self._filter_valid_file_metadatas(file_metadatas)
        return [f["id"] for f in valid_fname_objs]

    def owner_download_all_raw_events_from_syftbox(self) -> list[bytes]:
        """Download all event files from /SyftBox/myemail as raw bytes."""
        personal_syftbox_folder_id = self.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(personal_syftbox_folder_id)
        valid_fname_objs = self._get_valid_events_from_file_metadatas(file_metadatas)

        result = []
        for fname_obj in valid_fname_objs:
            gdrive_id = [
                f for f in file_metadatas if f["name"] == fname_obj.as_string()
            ][0]["id"]
            try:
                file_data = self.download_file(gdrive_id)
            except Exception as e:
                print(e)
                continue
            result.append(file_data)
        return result

    def owner_write_raw_bytes_to_outbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        outbox_folder_id = self._get_own_datasite_outbox_id(recipient)
        if outbox_folder_id is None:
            raise ValueError(f"Outbox folder for {recipient} not found")

        file_payload, _ = self.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [outbox_folder_id]}

        result = execute_with_retries(
            self.drive_service.files().create(
                body=file_metadata, media_body=file_payload, fields="id, parents"
            )
        )

        file_id = result.get("id")
        actual_parents = result.get("parents", [])
        if outbox_folder_id not in actual_parents:
            print(
                f"WARNING: Event message {file_id} was not placed in outbox folder "
                f"{outbox_folder_id}. Actual parents: {actual_parents}. "
                f"Moving file to correct folder..."
            )
            execute_with_retries(
                self.drive_service.files().update(
                    fileId=file_id,
                    addParents=outbox_folder_id,
                    fields="id, parents",
                )
            )

    def owner_write_event_messages_to_outbox(
        self, recipient: str, events_message: FileChangeEventsMessage
    ):
        data = events_message.as_compressed_data()
        fname = events_message.message_filepath.as_string()
        self.owner_write_raw_bytes_to_outbox(recipient, fname, data)

    def owner_remove_proposed_filechange_message_from_inbox(
        self, proposed_filechange_message: ProposedFileChangesMessage
    ):
        fname = proposed_filechange_message.message_filename.as_string()
        sender_email = proposed_filechange_message.sender_email

        # Use cached platform_id if available, otherwise fall back to name-based lookup
        gdrive_id = proposed_filechange_message.platform_id
        if gdrive_id is None:
            gdrive_id = self.get_inbox_proposed_event_id_from_name(sender_email, fname)
        if gdrive_id is None:
            raise ValueError(
                f"Event {fname} not found in inbox, event should already be created for this type of connection"
            )
        file_info = execute_with_retries(
            self.drive_service.files().get(fileId=gdrive_id, fields="parents")
        )
        previous_parents = ",".join(file_info.get("parents", []))
        archive_folder_id = self.owner_get_archive_folder_id(sender_email)
        execute_with_retries(
            self.drive_service.files().update(
                fileId=gdrive_id,
                addParents=archive_folder_id,
                removeParents=previous_parents,
                fields="id, parents",
                supportsAllDrives=True,
            )
        )

    def _has_permission(self, file_id: str, email: str) -> bool:
        """Check if user already has permission on the file."""
        perms = execute_with_retries(
            self.drive_service.permissions().list(
                fileId=file_id, fields="permissions(emailAddress)"
            )
        )
        for p in perms.get("permissions", []):
            if p.get("emailAddress", "").lower() == email.lower():
                return True
        return False

    def add_permission(self, file_id: str, recipient: str, write=False):
        """Add permission to the file if not already shared."""
        if self._has_permission(file_id, recipient):
            return

        role = "writer" if write else "reader"
        permission = {
            "type": "user",
            "role": role,
            "emailAddress": recipient,
        }
        execute_with_retries(
            self.drive_service.permissions().create(
                fileId=file_id, body=permission, sendNotificationEmail=True
            )
        )

    def _create_peer_datasite_folder(self, peer_email: str, folder_type: str) -> str:
        """Create a datasite folder under /SyftBox for peer's datasite."""
        if folder_type not in ("inbox", "outbox"):
            raise ValueError(
                f"Invalid folder_type: {folder_type}. Must be 'inbox' or 'outbox'."
            )
        parent_id = self.get_syftbox_folder_id()
        folder = GdriveP2PFolder(
            datasite_email=peer_email, folder_type=folder_type, peer_email=self.email
        )
        folder_name = folder.as_string()
        folder_id = self.create_folder(folder_name, parent_id)
        if folder_type == "inbox":
            self.peer_datasite_inbox_cache[peer_email] = folder_id
        else:
            self.peer_datasite_outbox_cache[peer_email] = folder_id
        return folder_id

    def get_personal_syftbox_folder_id(self) -> str:
        """/SyftBox/{version}#{email}"""
        if self._personal_syftbox_folder_id:
            return self._personal_syftbox_folder_id
        folders = self._find_folders(
            name_contains=[f"#{self.email}"],
            parent_id=self.get_syftbox_folder_id(),
            owner_email=self.email,
        )
        # The substring '#{email}' also matches p2p folder names that end in
        # '#{peer}#{type}#{email}'. Personal folder shape is exactly
        # '{version}#{email}', so require a single '#'.
        folders = [(fid, name) for fid, name in folders if name.count("#") == 1]
        folder_id = self._expect_one(_filter_patch_compatible(folders))
        if folder_id:
            self._personal_syftbox_folder_id = folder_id
            return folder_id
        return self.create_personal_syftbox_folder()

    def get_syftbox_folder_id(self) -> str:
        """/SyftBox"""
        # cached
        if self._syftbox_folder_id:
            return self._syftbox_folder_id
        else:
            syftbox_folder_id = self.get_syftbox_folder_id_from_drive()
            if syftbox_folder_id:
                self._syftbox_folder_id = syftbox_folder_id
                return self._syftbox_folder_id
            else:
                return self.create_syftbox_folder()

    def get_archive_folder_id_from_drive(self, sender_email: str) -> str | None:
        archive_folder = GdriveArchiveFolder(
            sender_email=sender_email, recipient_email=self.email
        )
        archive_folder_name = archive_folder.as_string()
        query = f"name='{archive_folder_name}' and mimeType='application/vnd.google-apps.folder' and 'me' in owners and trashed=false"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def owner_get_archive_folder_id(self, sender_email: str) -> str:
        if sender_email in self.archive_folder_id_cache:
            return self.archive_folder_id_cache[sender_email]
        else:
            archive_folder_id = self.get_archive_folder_id_from_drive(sender_email)
            if archive_folder_id:
                self.archive_folder_id_cache[sender_email] = archive_folder_id
                return archive_folder_id
            else:
                return self.create_archive_folder(sender_email)

    @staticmethod
    def _extract_timestamp_from_filename(filename: str) -> float | None:
        """
        Extract timestamp from filename.

        Supports multiple filename formats:
        - Event files: syfteventsmessagev3_<timestamp>_<uuid>.tar.gz
        - Job files: msgv2_<timestamp>_<uid>.tar.gz

        Args:
            filename: The filename to parse

        Returns:
            Timestamp as float, or None if can't parse
        """
        try:
            # Try event file format first
            if filename.startswith("syfteventsmessagev3_"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    return float(parts[1])

            # Try job file format
            if filename.startswith("msgv2_"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    return float(parts[1])

            return None
        except (ValueError, IndexError):
            return None

    def get_file_metadatas_from_folder(
        self,
        folder_id: str,
        since_timestamp: float | None = None,
        page_size: int = 100,
    ) -> List[Dict]:
        """
        Get file metadatas from folder with early termination.

        Args:
            folder_id: Google Drive folder ID
            since_timestamp: Optional timestamp. If provided, stops pagination
                           when encountering files with timestamp <= this value.
                           Enables early termination optimization.
            page_size: Number of files to fetch per API call. Default 100.

        Returns:
            List of file metadata dicts, sorted by name descending (newest first)
        """
        query = f"'{folder_id}' in parents and trashed=false"
        all_files = []
        page_token = None

        while True:
            results = execute_with_retries(
                self.drive_service.files().list(
                    q=query,
                    fields="files(id, name, size, mimeType, modifiedTime), nextPageToken",
                    pageSize=page_size,
                    pageToken=page_token,
                    orderBy="name desc",
                )
            )

            page_files = results.get("files", [])

            # Early termination: Check if this page contains old files
            if since_timestamp is not None and page_files:
                should_stop = False

                for file in page_files:
                    timestamp = self._extract_timestamp_from_filename(file["name"])

                    if timestamp is not None:
                        if timestamp > since_timestamp:
                            all_files.append(file)
                        else:
                            # Found a file we already have! Stop pagination
                            should_stop = True
                            break
                    else:
                        # No timestamp in filename, include the file
                        all_files.append(file)

                if should_stop:
                    # Don't fetch more pages
                    break
            else:
                # No early termination check, add all files
                all_files.extend(page_files)

            # Check for next page
            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return all_files

    @staticmethod
    def _filter_valid_file_metadatas(
        file_metadatas: List[Dict],
    ) -> List[Dict]:
        res = []
        for file_metadata in file_metadatas:
            fname = file_metadata["name"]
            try:
                _ = FileChangeEventsMessageFileName.from_string(fname)
                res.append(file_metadata)
            except Exception:
                continue
        return res

    @staticmethod
    def _get_valid_events_from_file_metadatas(
        file_metadatas: List[Dict],
    ) -> List[FileChangeEventsMessageFileName]:
        res = []
        for file_metadata in file_metadatas:
            fname = file_metadata["name"]
            try:
                message_filename = FileChangeEventsMessageFileName.from_string(fname)
                res.append(message_filename)
            except Exception:
                print("Warning, invalid file name: ", fname)
                continue
        return res

    @staticmethod
    def _get_valid_messages_from_file_metadatas(
        file_metadatas: List[Dict],
    ) -> List[MessageFileName]:
        res = []
        for file_metadata in file_metadatas:
            try:
                message_filename = MessageFileName.from_string(file_metadata["name"])
                res.append(message_filename)
            except FileNameParseError:
                continue
        return res

    def owner_download_next_raw_proposed_message_from_inbox(
        self, sender_email: str
    ) -> tuple[bytes, str] | None:
        inbox_folder_id = self._get_own_datasite_inbox_id(sender_email)
        if inbox_folder_id is None:
            raise ValueError(f"Inbox folder for {sender_email} not found")
        file_metadatas = self.get_file_metadatas_from_folder(inbox_folder_id)
        valid_file_names = self._get_valid_messages_from_file_metadatas(file_metadatas)
        if len(valid_file_names) == 0:
            return None
        first_file_name = sorted(valid_file_names, key=lambda x: x.submitted_timestamp)[
            0
        ]
        first_file_id = [
            x for x in file_metadatas if x["name"] == first_file_name.as_string()
        ][0]["id"]
        raw_data = self.download_file(first_file_id)
        return raw_data, first_file_id

    def owner_get_next_proposed_filechange_message(
        self, sender_email: str
    ) -> ProposedFileChangesMessage | None:
        result = self.owner_download_next_raw_proposed_message_from_inbox(sender_email)
        if result is None:
            return None
        raw_data, file_id = result
        msg = ProposedFileChangesMessage.from_compressed_data(raw_data)
        msg.platform_id = file_id
        return msg

    def _find_p2p_folder_id(
        self,
        datasite_email: str,
        folder_type: str,
        peer_email: str,
        owner_email: str,
    ) -> str | None:
        folders = self._find_folders(
            name_contains=[
                f"{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#",
                f"#{datasite_email}#{folder_type}#{peer_email}",
            ],
            owner_email=owner_email,
        )

        # Drive's `name contains` is a fuzzy token/prefix match, not a substring
        # match, so it can return folders for other peers when email tokens
        # collide
        def _is_exact_match(name: str) -> bool:
            try:
                folder = GdriveP2PFolder.from_name(name)
            except ValueError:
                return False
            return (
                folder.datasite_email == datasite_email
                and folder.folder_type == folder_type
                and folder.peer_email == peer_email
            )

        folders = [(fid, name) for fid, name in folders if _is_exact_match(name)]
        return self._expect_one(_filter_patch_compatible(folders))

    def _get_peer_datasite_inbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{peer}_inbox_{self}, owned by self."""
        if peer_email in self.peer_datasite_inbox_cache:
            return self.peer_datasite_inbox_cache[peer_email]
        folder_id = self._find_p2p_folder_id(
            datasite_email=peer_email,
            folder_type="inbox",
            peer_email=self.email,
            owner_email=self.email,
        )
        if folder_id is not None:
            self.peer_datasite_inbox_cache[peer_email] = folder_id
        return folder_id

    def _get_peer_datasite_outbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{peer}_outbox_{self}, owned by self."""
        if peer_email in self.peer_datasite_outbox_cache:
            return self.peer_datasite_outbox_cache[peer_email]
        folder_id = self._find_p2p_folder_id(
            datasite_email=peer_email,
            folder_type="outbox",
            peer_email=self.email,
            owner_email=self.email,
        )
        if folder_id is not None:
            self.peer_datasite_outbox_cache[peer_email] = folder_id
        return folder_id

    def _get_own_datasite_inbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{self}_inbox_{peer}, owned by peer."""
        if peer_email in self.own_datasite_inbox_cache:
            return self.own_datasite_inbox_cache[peer_email]
        folder_id = self._find_p2p_folder_id(
            datasite_email=self.email,
            folder_type="inbox",
            peer_email=peer_email,
            owner_email=peer_email,
        )
        if folder_id is not None:
            self.own_datasite_inbox_cache[peer_email] = folder_id
        return folder_id

    def _get_own_datasite_outbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{self}_outbox_{peer}, owned by peer."""
        if peer_email in self.own_datasite_outbox_cache:
            return self.own_datasite_outbox_cache[peer_email]
        folder_id = self._find_p2p_folder_id(
            datasite_email=self.email,
            folder_type="outbox",
            peer_email=peer_email,
            owner_email=peer_email,
        )
        if folder_id is not None:
            self.own_datasite_outbox_cache[peer_email] = folder_id
        return folder_id

    def watcher_send_raw_bytes_to_inbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        inbox_id = self._get_peer_datasite_inbox_id(recipient)
        if inbox_id is None:
            raise Exception(f"Inbox folder for {recipient}'s datasite not found")

        payload, _ = self.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [inbox_id]}

        result = execute_with_retries(
            self.drive_service.files().create(
                body=file_metadata, media_body=payload, fields="id, parents"
            )
        )

        file_id = result.get("id")
        actual_parents = result.get("parents", [])
        if inbox_id not in actual_parents:
            print(
                f"WARNING: Message file {file_id} was not placed in inbox folder "
                f"{inbox_id}. Actual parents: {actual_parents}. "
                f"Moving file to correct folder..."
            )
            execute_with_retries(
                self.drive_service.files().update(
                    fileId=file_id,
                    addParents=inbox_id,
                    fields="id, parents",
                )
            )

    def watcher_send_proposed_file_changes_message(
        self,
        recipient: str,
        proposed_file_changes_message: ProposedFileChangesMessage,
    ):
        data = proposed_file_changes_message.as_compressed_data()
        filename = proposed_file_changes_message.message_filename.as_string()
        self.watcher_send_raw_bytes_to_inbox(recipient, filename, data)

    def reset_caches(self):
        self._syftbox_folder_id = None
        self._personal_syftbox_folder_id = None
        self.peer_datasite_inbox_cache.clear()
        self.peer_datasite_outbox_cache.clear()
        self.own_datasite_inbox_cache.clear()
        self.own_datasite_outbox_cache.clear()
        self.archive_folder_id_cache.clear()
        self.personal_syftbox_event_id_cache.clear()
        self.dataset_collection_folder_id_cache.clear()
        self._rolling_state_folder_id = None
        self._rolling_state_file_id = None
        self._encryption_bundles_folder_id = None
        self._peers_json_cache = None

    def gather_all_file_and_folder_ids(self) -> List[str]:
        syftbox_folder_id = self.get_syftbox_folder_id()
        return gather_all_file_and_folder_ids_recursive(
            self.drive_service, syftbox_folder_id
        )

    def delete_multiple_files_by_ids(
        self,
        file_ids: List[str],
        ignore_permissions_errors: bool = True,
        ignore_file_not_found: bool = True,
    ):
        def callback(request_id, response, exception):
            if exception:
                exception_str = str(exception)
                # insufficientFilePermissions is a common error when deleting files that may already be removed
                if (
                    ignore_permissions_errors
                    and "insufficientFilePermissions" in exception_str
                ):
                    return
                # 404 errors occur when files are already deleted
                if ignore_file_not_found and (
                    "404" in exception_str or "notFound" in exception_str
                ):
                    return
                raise exception
            if (
                response is not None
                and not isinstance(response, str)
                and response.get("status")
                and int(response.get("status")) >= 400
            ):
                raise Exception(
                    f"Failed to delete {request_id}: error status {response.get('status')}"
                )

        # Google Drive batch API has a limit of 100 requests per batch
        BATCH_SIZE = 100
        for i in range(0, len(file_ids), BATCH_SIZE):
            chunk = file_ids[i : i + BATCH_SIZE]
            batch = self.drive_service.new_batch_http_request(callback=callback)
            for file_id in chunk:
                batch.add(self.drive_service.files().delete(fileId=file_id))
            batch_execute_with_retries(batch)

    def delete_file_by_id(
        self, file_id: str, verbose: bool = False, raise_on_error: bool = False
    ):
        try:
            execute_with_retries(self.drive_service.files().delete(fileId=file_id))
        except Exception as e:
            if raise_on_error:
                raise e
            else:
                if verbose:
                    print(f"Error deleting file: {file_id}")

    def delete_unversioned_state(self) -> None:
        """Delete non-versioned remote artifacts during upgrade.

        Removes encryption bundles, dataset collections, private collections,
        peers file, and version file from /SyftBox/.
        """
        syftbox_folder_id = self.get_syftbox_folder_id()
        ids_to_delete: list[str] = []

        # 1. Encryption bundles folder
        enc_folder_name = GdriveEncryptionBundlesFolder(email=self.email).as_string()
        enc_folder_id = self._find_folder_by_name(
            enc_folder_name, parent_id=syftbox_folder_id
        )
        if enc_folder_id:
            ids_to_delete.extend(
                gather_all_file_and_folder_ids_recursive(
                    self.drive_service, enc_folder_id
                )
            )
            ids_to_delete.append(enc_folder_id)

        # 2. Dataset collection folders (syft_datasetcollection_*)
        ds_query = (
            f"name contains '{DATASET_COLLECTION_PREFIX}'"
            f" and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
            f" and '{syftbox_folder_id}' in parents"
            " and trashed=false"
        )
        ds_results = execute_with_retries(
            self.drive_service.files().list(q=ds_query, fields="files(id)")
        )
        for f in ds_results.get("files", []):
            ids_to_delete.extend(
                gather_all_file_and_folder_ids_recursive(self.drive_service, f["id"])
            )
            ids_to_delete.append(f["id"])

        # 3. Private collection folders (syft_privatecollection_*)
        pc_query = (
            f"name contains '{PRIVATE_DATASET_COLLECTION_PREFIX}'"
            f" and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
            f" and '{syftbox_folder_id}' in parents"
            " and trashed=false"
        )
        pc_results = execute_with_retries(
            self.drive_service.files().list(q=pc_query, fields="files(id)")
        )
        for f in pc_results.get("files", []):
            ids_to_delete.extend(
                gather_all_file_and_folder_ids_recursive(self.drive_service, f["id"])
            )
            ids_to_delete.append(f["id"])

        # 4. SYFT_peers.json
        peers_file_id = self._get_peers_file_id()
        if peers_file_id:
            ids_to_delete.append(peers_file_id)

        # 5. SYFT_version.json
        version_file_id = self._get_version_file_id()
        if version_file_id:
            ids_to_delete.append(version_file_id)

        if ids_to_delete:
            self.delete_multiple_files_by_ids(ids_to_delete)
            self.reset_caches()

    def find_orphaned_message_files(self) -> list[str]:
        """
        Find syft files by name pattern owned by user, regardless of parent folder.

        Due to Google Drive's eventual consistency, files can become orphaned when
        their parent folder is deleted before they're fully registered. This method
        finds such files by searching for name patterns regardless of parent.

        Returns list of file IDs.
        """
        patterns = [
            "syfteventsmessagev3_",  # event messages
            "msgv2_",  # proposed file change messages
            CHECKPOINT_FILENAME_PREFIX,  # checkpoint and incremental checkpoint files
            ROLLING_STATE_FILENAME_PREFIX,  # rolling state files
            GDRIVE_P2P_FOLDER_DATASITE_PREFIX,  # new: syft_datasite_ folders
            LEGACY_GDRIVE_OUTBOX_INBOX_FOLDER_PREFIX,  # legacy: syft_outbox_inbox folders
            "syft_encryption_bundles",  # encryption bundles folders
            "encryption_bundle_",  # encryption bundle files
        ]
        file_ids = []

        for pattern in patterns:
            query = f"name contains '{pattern}' and 'me' in owners and trashed=false"
            page_token = None

            while True:
                results = execute_with_retries(
                    self.drive_service.files().list(
                        q=query,
                        fields="files(id), nextPageToken",
                        pageToken=page_token,
                    )
                )

                for item in results.get("files", []):
                    file_ids.append(item["id"])

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

        return file_ids

    def create_file_payload(self, data: Any) -> Tuple[MediaIoBaseUpload, str]:
        """Create a file payload for the GDrive"""
        if isinstance(data, str):
            file_data = data.encode("utf-8")
            mime_type = "text/plain"
            extension = ".txt"
        elif isinstance(data, dict):
            file_data = json.dumps(data, indent=2).encode("utf-8")
            mime_type = "application/json"
            extension = ".json"
        elif isinstance(data, bytes):
            file_data = data
            mime_type = "application/octet-stream"
            extension = ".bin"
        else:
            # Pickle for other data types
            file_data = pickle.dumps(data)
            mime_type = "application/octet-stream"
            extension = ".pkl"

        media = MediaIoBaseUpload(
            io.BytesIO(file_data), mimetype=mime_type, resumable=True
        )

        return media, extension

    def _find_folder_by_name(
        self, folder_name: str, parent_id: str = None, owner_email: str = None
    ) -> Optional[str]:
        """Find a folder by name, optionally within a specific parent"""
        # parent_id = "1AQ3WLnVlLd6Zjo7p9Z_qGA1Djjf6-KIh"
        owner_email_clause = f"and '{owner_email}' in owners" if owner_email else ""
        parent_id_clause = f"and '{parent_id}' in parents" if parent_id else ""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false {owner_email_clause} {parent_id_clause}"

        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)", pageSize=1)
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def _find_folders(
        self,
        *,
        name_contains: list[str] = (),
        parent_id: str | None = None,
        owner_email: str | None = None,
    ) -> list[tuple[str, str]]:
        """List folders matching the constraints. Returns (id, name) pairs.

        Thin wrapper over Drive's files.list -- handles query building and
        pagination, knows nothing about versions. Pair with
        _filter_patch_compatible when the caller cares about version compat.
        """
        clauses = [f"mimeType='{GOOGLE_FOLDER_MIME_TYPE}'", "trashed=false"]
        for substr in name_contains:
            clauses.append(f"name contains '{substr}'")
        if parent_id:
            clauses.append(f"'{parent_id}' in parents")
        if owner_email:
            clauses.append(f"'{owner_email}' in owners")
        query = " and ".join(clauses)

        out: list[tuple[str, str]] = []
        page_token = None
        while True:
            page = execute_with_retries(
                self.drive_service.files().list(
                    q=query,
                    fields="files(id, name), nextPageToken",
                    pageToken=page_token,
                )
            )
            out.extend((f["id"], f["name"]) for f in page.get("files", []))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        return out

    def _expect_one(self, folders: list[tuple[str, str]]) -> str | None:
        """Return the single folder id, None if empty, raise if more than one.

        Auto-picking from multiple matches risks orphaning data, so the user
        must reconcile on Drive themselves.
        """
        if not folders:
            return None
        if len(folders) == 1:
            return folders[0][0]
        names = [n for _, n in folders]
        raise RuntimeError(
            f"Found {len(folders)} compatible folders on Drive: {names}. "
            f"Exactly one is expected. This usually indicates a stale folder "
            f"left behind by a prior client version. Please delete the stale "
            f"folder(s) on Drive (keeping the one with your data) and retry."
        )

    def download_file(self, file_id: str) -> bytes:
        request = self.drive_service.files().get_media(fileId=file_id)

        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(
            file_buffer, request, chunksize=1024 * 1024 * 10
        )

        done = False
        while not done:
            status, done = next_chunk_with_retries(downloader)

        message_data = file_buffer.getvalue()
        return message_data

    def create_folder(self, folder_name: str, parent_id: str) -> str:
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            file_metadata["parents"] = [parent_id]
        folder = execute_with_retries(
            self.drive_service.files().create(body=file_metadata, fields="id")
        )
        return folder.get("id")

    def get_syftbox_folder_id_from_drive(self) -> str | None:
        query = f"name='{SYFTBOX_FOLDER}' and mimeType='application/vnd.google-apps.folder' and 'me' in owners and trashed=false"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def get_inbox_proposed_event_id_from_name(
        self, sender_email: str, name: str
    ) -> str | None:
        inbox_folder_id = self._get_own_datasite_inbox_id(sender_email)
        query = f"name='{name}' and '{inbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def owner_create_dataset_collection_folder(
        self, tag: str, content_hash: str, owner_email: str
    ) -> str:
        """Create /SyftBox/{DATASET_COLLECTION_PREFIX}_{tag}_{hash} folder."""
        folder_obj = DatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        cache_key = f"{tag}_{content_hash}"

        # Check cache
        if cache_key in self.dataset_collection_folder_id_cache:
            return self.dataset_collection_folder_id_cache[cache_key]

        syftbox_folder_id = self.get_syftbox_folder_id()

        # Check if exists
        folder_id = self._find_folder_by_name(folder_name, parent_id=syftbox_folder_id)
        if folder_id:
            self.dataset_collection_folder_id_cache[cache_key] = folder_id
            return folder_id

        # Create new folder
        folder_id = self.create_folder(folder_name, syftbox_folder_id)
        self.dataset_collection_folder_id_cache[cache_key] = folder_id
        return folder_id

    def owner_tag_dataset_collection_as_any(self, tag: str, content_hash: str) -> None:
        """Mark dataset collection as shared with 'any' via appProperties."""
        folder_id = self._get_dataset_collection_folder_id(tag, content_hash)
        execute_with_retries(
            self.drive_service.files().update(
                fileId=folder_id,
                body={"appProperties": {"syft_shared_with_any": "true"}},
            )
        )

    def owner_share_dataset_collection(
        self, tag: str, content_hash: str, users: list[str]
    ) -> None:
        """Share dataset collection folder with specific users via batch API."""
        if not users:
            return
        folder_id = self._get_dataset_collection_folder_id(tag, content_hash)
        self._batch_add_permissions(folder_id, users)

    def _batch_add_permissions(self, file_id: str, users: list[str]) -> None:
        """Add reader permissions for multiple users in a single batch request."""

        def callback(request_id, response, exception):
            if exception:
                # Ignore "already shared" errors
                if "alreadyShared" not in str(exception):
                    raise exception

        BATCH_SIZE = 100
        for i in range(0, len(users), BATCH_SIZE):
            chunk = users[i : i + BATCH_SIZE]
            batch = self.drive_service.new_batch_http_request(callback=callback)
            for user_email in chunk:
                permission = {
                    "type": "user",
                    "role": "reader",
                    "emailAddress": user_email,
                }
                batch.add(
                    self.drive_service.permissions().create(
                        fileId=file_id,
                        body=permission,
                        sendNotificationEmail=True,
                    )
                )
            batch_execute_with_retries(batch)

    def owner_upload_dataset_files(
        self, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        """Upload dataset files to collection folder."""
        folder_id = self._get_dataset_collection_folder_id(tag, content_hash)

        for file_path, content in files.items():
            file_payload, _ = self.create_file_payload(content)
            file_name = Path(file_path).name

            file_metadata = {"name": file_name, "parents": [folder_id]}
            execute_with_retries(
                self.drive_service.files().create(
                    body=file_metadata, media_body=file_payload, fields="id"
                )
            )

    def owner_list_dataset_collections(self) -> list[str]:
        """List collections created by DO (owned by me)."""
        syftbox_folder_id = self.get_syftbox_folder_id()
        query = (
            f"name contains '{DATASET_COLLECTION_PREFIX}_' and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(name)")
        )

        folders = results.get("files", [])
        result = []
        for folder in folders:
            try:
                folder_obj = DatasetCollectionFolder.from_name(folder["name"])
                result.append(folder_obj.tag)
            except ValueError:
                continue
        return result

    def owner_list_all_dataset_collections_with_permissions(
        self,
    ) -> list[FileCollection]:
        """List all DO's dataset collections with permissions info."""
        syftbox_folder_id = self.get_syftbox_folder_id()
        query = (
            f"name contains '{DATASET_COLLECTION_PREFIX}_' and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        results = execute_with_retries(
            self.drive_service.files().list(
                q=query, fields="files(id,name,appProperties)"
            )
        )

        collections = []
        for folder in results.get("files", []):
            folder_id = folder["id"]
            try:
                folder_obj = DatasetCollectionFolder.from_name(folder["name"])
                has_anyone = (
                    folder.get("appProperties", {}).get("syft_shared_with_any")
                    == "true"
                )
                collections.append(
                    FileCollection(
                        folder_id=folder_id,
                        tag=folder_obj.tag,
                        content_hash=folder_obj.content_hash,
                        has_any_permission=has_anyone,
                    )
                )
            except Exception:
                continue

        return collections

    def owner_delete_dataset_collection(self, tag: str) -> None:
        """Delete all public dataset collection folders matching the given tag."""
        collections = self.owner_list_all_dataset_collections_with_permissions()
        for c in collections:
            if c.tag == tag:
                self.delete_file_by_id(c.folder_id)
                cache_key = f"{c.tag}_{c.content_hash}"
                self.dataset_collection_folder_id_cache.pop(cache_key, None)

    def watcher_list_dataset_collections(self) -> list[dict]:
        """List collections shared with DS (not owned by me).

        Returns list of dicts with keys: owner_email, tag, content_hash
        """
        query = (
            f"name contains '{DATASET_COLLECTION_PREFIX}_' and not 'me' in owners "
            f"and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(name, owners)")
        )

        folders = results.get("files", [])
        result = []
        for folder in folders:
            try:
                folder_obj = DatasetCollectionFolder.from_name(folder["name"])
                owner_email = folder.get("owners", [{}])[0].get(
                    "emailAddress", "unknown"
                )
                result.append(
                    {
                        "owner_email": owner_email,
                        "tag": folder_obj.tag,
                        "content_hash": folder_obj.content_hash,
                    }
                )
            except ValueError:
                # Skip folders that don't match the expected format
                continue
        return result

    def watcher_download_dataset_collection(
        self, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        """Download all files from a dataset collection."""
        folder_obj = DatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        # Try to find folder by name (could be owned by someone else)
        folder_id = self._find_folder_by_name(folder_name, owner_email=owner_email)

        if not folder_id:
            raise ValueError(f"Collection {tag} with hash {content_hash} not found")

        file_metadatas = self.get_file_metadatas_from_folder(folder_id)
        files = {}
        for file_meta in file_metadatas:
            file_id = file_meta["id"]
            file_name = file_meta["name"]
            files[file_name] = self.download_file(file_id)

        return files

    def watcher_get_dataset_collection_file_metadatas(
        self, tag: str, content_hash: str, owner_email: str
    ) -> List[Dict]:
        """Get file metadata from a dataset collection without downloading."""
        folder_obj = DatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        folder_id = self._find_folder_by_name(folder_name, owner_email=owner_email)

        if not folder_id:
            raise ValueError(f"Collection {tag} with hash {content_hash} not found")

        file_metadatas = self.get_file_metadatas_from_folder(folder_id)
        return [{"file_id": f["id"], "file_name": f["name"]} for f in file_metadatas]

    def watcher_download_dataset_file(self, file_id: str) -> bytes:
        """Download a single file from a dataset collection."""
        return self.download_file(file_id)

    def _get_dataset_collection_folder_id(self, tag: str, content_hash: str) -> str:
        """Get folder ID for dataset collection, with caching."""
        cache_key = f"{tag}_{content_hash}"
        if cache_key in self.dataset_collection_folder_id_cache:
            return self.dataset_collection_folder_id_cache[cache_key]

        folder_obj = DatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_id = self._find_folder_by_name(folder_name, parent_id=syftbox_folder_id)

        if not folder_id:
            raise ValueError(
                f"Collection folder {tag} with hash {content_hash} not found"
            )

        self.dataset_collection_folder_id_cache[cache_key] = folder_id
        return folder_id

    # =========================================================================
    # PRIVATE DATASET COLLECTION METHODS
    # =========================================================================

    def owner_create_private_dataset_collection_folder(
        self, tag: str, content_hash: str, owner_email: str
    ) -> str:
        """Create /SyftBox/{PRIVATE_DATASET_COLLECTION_PREFIX}_{tag}_{hash} folder.

        No sharing is applied — only the owner can access this folder.
        """
        folder_obj = PrivateDatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        cache_key = f"private_{tag}_{content_hash}"

        if cache_key in self.dataset_collection_folder_id_cache:
            return self.dataset_collection_folder_id_cache[cache_key]

        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_id = self._find_folder_by_name(folder_name, parent_id=syftbox_folder_id)
        if folder_id:
            self.dataset_collection_folder_id_cache[cache_key] = folder_id
            return folder_id

        folder_id = self.create_folder(folder_name, syftbox_folder_id)
        self.dataset_collection_folder_id_cache[cache_key] = folder_id
        return folder_id

    def owner_upload_private_dataset_files(
        self, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        """Upload files to a private dataset collection folder."""
        folder_id = self._get_private_collection_folder_id(tag, content_hash)
        for file_path, content in files.items():
            file_payload, _ = self.create_file_payload(content)
            file_name = Path(file_path).name
            file_metadata = {"name": file_name, "parents": [folder_id]}
            execute_with_retries(
                self.drive_service.files().create(
                    body=file_metadata, media_body=file_payload, fields="id"
                )
            )

    def owner_list_private_dataset_collections(self) -> list[FileCollection]:
        """List private collections owned by DO."""
        syftbox_folder_id = self.get_syftbox_folder_id()
        query = (
            f"name contains '{PRIVATE_DATASET_COLLECTION_PREFIX}_' "
            f"and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false "
            f"and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id,name)")
        )

        collections = []
        for folder in results.get("files", []):
            try:
                folder_obj = PrivateDatasetCollectionFolder.from_name(folder["name"])
                collections.append(
                    FileCollection(
                        folder_id=folder["id"],
                        tag=folder_obj.tag,
                        content_hash=folder_obj.content_hash,
                    )
                )
            except ValueError:
                continue
        return collections

    def owner_delete_private_dataset_collection(self, tag: str) -> None:
        """Delete all private dataset collection folders matching the given tag."""
        collections = self.owner_list_private_dataset_collections()
        for c in collections:
            if c.tag == tag:
                self.delete_file_by_id(c.folder_id)
                cache_key = f"private_{c.tag}_{c.content_hash}"
                self.dataset_collection_folder_id_cache.pop(cache_key, None)

    def owner_get_private_collection_file_metadatas(
        self, tag: str, content_hash: str, owner_email: str
    ) -> List[Dict]:
        """Get file metadata from a private dataset collection without downloading."""
        folder_obj = PrivateDatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        folder_id = self._find_folder_by_name(folder_name, owner_email=owner_email)

        if not folder_id:
            raise ValueError(
                f"Private collection {tag} with hash {content_hash} not found"
            )

        file_metadatas = self.get_file_metadatas_from_folder(folder_id)
        return [{"file_id": f["id"], "file_name": f["name"]} for f in file_metadatas]

    def _get_private_collection_folder_id(self, tag: str, content_hash: str) -> str:
        """Get folder ID for private dataset collection, with caching."""
        cache_key = f"private_{tag}_{content_hash}"
        if cache_key in self.dataset_collection_folder_id_cache:
            return self.dataset_collection_folder_id_cache[cache_key]

        folder_obj = PrivateDatasetCollectionFolder(tag=tag, content_hash=content_hash)
        folder_name = folder_obj.as_string()
        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_id = self._find_folder_by_name(folder_name, parent_id=syftbox_folder_id)

        if not folder_id:
            raise ValueError(
                f"Private collection folder {tag} with hash {content_hash} not found"
            )

        self.dataset_collection_folder_id_cache[cache_key] = folder_id
        return folder_id

    def _get_version_file_id(self) -> Optional[str]:
        """Find SYFT_version.json file in /SyftBox folder"""
        syftbox_folder_id = self.get_syftbox_folder_id()
        query = f"name='{SYFT_VERSION_FILE}' and '{syftbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def write_version_file(self, version_info: "VersionInfo") -> None:
        """Write version file to /SyftBox folder. Creates or updates the file."""

        syftbox_folder_id = self.get_syftbox_folder_id()
        file_id = self._get_version_file_id()

        # Convert to JSON string
        json_data = version_info.to_json()
        file_payload, _ = self.create_file_payload(json_data)

        if file_id is None:
            # Create new file
            file_metadata = {
                "name": SYFT_VERSION_FILE,
                "parents": [syftbox_folder_id],
            }
            execute_with_retries(
                self.drive_service.files().create(
                    body=file_metadata, media_body=file_payload, fields="id"
                )
            )
        else:
            # Update existing file
            execute_with_retries(
                self.drive_service.files().update(
                    fileId=file_id, media_body=file_payload
                )
            )

    def _get_peer_version_file_id(self, peer_email: str) -> Optional[str]:
        """Find SYFT_version.json file in a peer's /SyftBox folder"""
        # Find the peer's SyftBox folder
        query = (
            f"name='{SYFT_VERSION_FILE}' and trashed=false and '{peer_email}' in owners"
        )
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def read_own_version_file(self) -> Optional["VersionInfo"]:
        """Read version file from own /SyftBox folder."""
        from syft_client.sync.version.version_info import VersionInfo

        file_id = self._get_version_file_id()
        if file_id is None:
            return None

        try:
            file_data = self.download_file(file_id)
            return VersionInfo.from_json(file_data.decode("utf-8"))
        except Exception:
            return None

    def read_peer_version_file(self, peer_email: str) -> Optional["VersionInfo"]:
        """Read version file from a peer's /SyftBox folder."""
        from syft_client.sync.version.version_info import VersionInfo

        file_id = self._get_peer_version_file_id(peer_email)
        if file_id is None:
            return None

        try:
            file_data = self.download_file(file_id)
            return VersionInfo.from_json(file_data.decode("utf-8"))
        except Exception:
            return None

    def share_version_file_with_peer(self, peer_email: str) -> None:
        """Share the version file with a peer so they can read it."""
        file_id = self._get_version_file_id()
        if file_id is None:
            # Version file doesn't exist yet, create it first
            from syft_client.sync.version.version_info import VersionInfo

            self.write_version_file(VersionInfo.current())
            file_id = self._get_version_file_id()

        if file_id:
            self.add_permission(file_id, peer_email, write=False)

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def _get_checkpoints_folder_name(self) -> str:
        """Get the checkpoints folder name: {email}-{version}-checkpoints"""
        return f"{self.email}-{SYFT_CLIENT_VERSION}-checkpoints"

    def _get_checkpoints_folder_id(self) -> str | None:
        """Find the checkpoints folder ID from Google Drive."""
        folders = self._find_folders(
            name_contains=[f"{self.email}-", "-checkpoints"],
            parent_id=self.get_syftbox_folder_id(),
        )
        return self._expect_one(_filter_patch_compatible(folders))

    def _get_or_create_checkpoints_folder_id(self) -> str:
        """Get or create the checkpoints folder."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is not None:
            return folder_id
        # Create the folder
        folder_name = self._get_checkpoints_folder_name()
        syftbox_folder_id = self.get_syftbox_folder_id()
        return self.create_folder(folder_name, syftbox_folder_id)

    def upload_raw_checkpoint(self, filename: str, data: bytes) -> str:
        """Upload raw checkpoint bytes to Google Drive.

        Uploads the new checkpoint first, then deletes old ones.
        """
        folder_id = self._get_or_create_checkpoints_folder_id()
        payload, _ = self.create_file_payload(data)

        file_metadata = {
            "name": filename,
            "parents": [folder_id],
        }

        result = (
            self.drive_service.files()
            .create(body=file_metadata, media_body=payload, fields="id")
            .execute()
        )

        # Only delete old checkpoints after successful upload
        self.delete_all_checkpoints(exclude_file_id=result.get("id"))

        return result.get("id")

    def download_raw_latest_checkpoint(self) -> bytes | None:
        """Download the latest checkpoint as raw bytes, or None."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return None

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{CHECKPOINT_FILENAME_PREFIX}'"
        )
        results = (
            self.drive_service.files().list(q=query, fields="files(id, name)").execute()
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

        try:
            return self.download_file(latest_file["id"])
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def delete_all_checkpoints(self, exclude_file_id: str | None = None):
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
        results = (
            self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        )
        items = results.get("files", [])

        # Delete only full checkpoints (not incremental ones)
        for item in items:
            if item["name"].startswith(INCREMENTAL_CHECKPOINT_PREFIX):
                continue  # Skip incremental checkpoints
            if item["id"] == exclude_file_id:
                continue  # Skip the newly uploaded checkpoint
            try:
                self.drive_service.files().delete(fileId=item["id"]).execute()
            except Exception as e:
                print(f"Warning: Failed to delete checkpoint {item['name']}: {e}")

    # =========================================================================
    # INCREMENTAL CHECKPOINT METHODS
    # =========================================================================

    def upload_raw_incremental_checkpoint(self, filename: str, data: bytes) -> str:
        """Upload raw incremental checkpoint bytes to Google Drive."""
        folder_id = self._get_or_create_checkpoints_folder_id()
        payload, _ = self.create_file_payload(data)

        file_metadata = {
            "name": filename,
            "parents": [folder_id],
        }

        result = (
            self.drive_service.files()
            .create(body=file_metadata, media_body=payload, fields="id")
            .execute()
        )
        return result.get("id")

    def download_all_raw_incremental_checkpoints(self) -> list[bytes]:
        """Download all incremental checkpoints as raw bytes."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return []

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = (
            self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        )
        items = results.get("files", [])

        if not items:
            return []

        result = []
        for item in items:
            try:
                file_data = self.download_file(item["id"])
                result.append(file_data)
            except Exception as e:
                print(
                    f"Warning: Failed to load incremental checkpoint {item['name']}: {e}"
                )
                continue
        return result

    def get_incremental_checkpoint_count(self) -> int:
        """Get the number of incremental checkpoints on Google Drive."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return 0

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = self.drive_service.files().list(q=query, fields="files(id)").execute()
        return len(results.get("files", []))

    def get_next_incremental_sequence_number(self) -> int:
        """Get the next sequence number for incremental checkpoints."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return 1

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = (
            self.drive_service.files().list(q=query, fields="files(name)").execute()
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

    def delete_all_incremental_checkpoints(self) -> None:
        """Delete all incremental checkpoints (called after compacting)."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = (
            self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        )
        items = results.get("files", [])

        for item in items:
            try:
                self.drive_service.files().delete(fileId=item["id"]).execute()
            except Exception as e:
                print(
                    f"Warning: Failed to delete incremental checkpoint {item['name']}: {e}"
                )

    def get_events_count_since_checkpoint(
        self, checkpoint_timestamp: float | None
    ) -> int:
        """
        Count events created after the checkpoint timestamp.

        Args:
            checkpoint_timestamp: The timestamp of the checkpoint, or None for all events.

        Returns:
            Number of events since the checkpoint.
        """
        personal_folder_id = self.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(personal_folder_id)

        if checkpoint_timestamp is None:
            # Count all events
            return len(
                [
                    f
                    for f in file_metadatas
                    if f["name"].startswith("syfteventsmessagev3_")
                ]
            )

        # Count events after checkpoint
        count = 0
        for f in file_metadatas:
            if not f["name"].startswith("syfteventsmessagev3_"):
                continue
            event_timestamp = self._extract_timestamp_from_filename(f["name"])
            if event_timestamp is not None and event_timestamp > checkpoint_timestamp:
                count += 1
        return count

    def download_raw_events_since_timestamp(
        self, since_timestamp: float
    ) -> list[bytes]:
        """Download event files created after a timestamp as raw bytes."""
        personal_folder_id = self.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(
            personal_folder_id, since_timestamp=since_timestamp
        )

        valid_fname_objs = self._get_valid_events_from_file_metadatas(file_metadatas)

        result = []
        for fname_obj in valid_fname_objs:
            if fname_obj.timestamp <= since_timestamp:
                continue

            gdrive_id = [
                f for f in file_metadatas if f["name"] == fname_obj.as_string()
            ][0]["id"]

            try:
                file_data = self.download_file(gdrive_id)
                result.append(file_data)
            except Exception as e:
                print(f"Warning: Failed to download event: {e}")
                continue

        return result

    # =========================================================================
    # ROLLING STATE METHODS
    # =========================================================================

    def _get_rolling_state_folder_name(self) -> str:
        """Get the rolling state folder name: {email}-{version}-rolling-state"""
        return f"{self.email}-{SYFT_CLIENT_VERSION}-rolling-state"

    def _get_rolling_state_folder_id(self, use_cache: bool = True) -> str | None:
        """
        Find the rolling state folder ID from Google Drive.

        Args:
            use_cache: If True, return cached value if available.
        """
        if use_cache and self._rolling_state_folder_id is not None:
            return self._rolling_state_folder_id

        folders = self._find_folders(
            name_contains=[f"{self.email}-", "-rolling-state"],
            parent_id=self.get_syftbox_folder_id(),
        )
        folder_id = self._expect_one(_filter_patch_compatible(folders))
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
        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_id = self.create_folder(folder_name, syftbox_folder_id)
        self._rolling_state_folder_id = folder_id
        return folder_id

    def upload_raw_rolling_state(self, filename: str, data: bytes) -> str:
        """Upload raw rolling state bytes to Google Drive.

        Optimized to use update() if file ID is cached.
        """
        payload, _ = self.create_file_payload(data)

        # Try to update existing file if we have a cached ID
        if self._rolling_state_file_id is not None:
            try:
                self.drive_service.files().update(
                    fileId=self._rolling_state_file_id,
                    media_body=payload,
                ).execute()
                return self._rolling_state_file_id
            except Exception:
                self._rolling_state_file_id = None

        folder_id = self._get_or_create_rolling_state_folder_id()

        file_metadata = {
            "name": filename,
            "parents": [folder_id],
        }

        result = (
            self.drive_service.files()
            .create(body=file_metadata, media_body=payload, fields="id")
            .execute()
        )
        file_id = result.get("id")
        self._rolling_state_file_id = file_id
        return file_id

    def download_raw_rolling_state(self) -> bytes | None:
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
        results = (
            self.drive_service.files().list(q=query, fields="files(id, name)").execute()
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

        self._rolling_state_file_id = latest_file["id"]

        try:
            return self.download_file(latest_file["id"])
        except Exception as e:
            print(f"Warning: Failed to load rolling state: {e}")
            self._rolling_state_file_id = None
            return None

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
        results = (
            self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        )
        items = results.get("files", [])

        # Delete each rolling state file
        for item in items:
            try:
                self.drive_service.files().delete(fileId=item["id"]).execute()
            except Exception as e:
                print(f"Warning: Failed to delete rolling state {item['name']}: {e}")

    # =========================================================================
    # ENCRYPTION BUNDLE METHODS
    # =========================================================================

    def _get_or_create_encryption_bundles_folder_id(self) -> str:
        """Get or create the encryption bundles folder for this user."""
        if self._encryption_bundles_folder_id:
            return self._encryption_bundles_folder_id
        folder_name = GdriveEncryptionBundlesFolder(email=self.email).as_string()
        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_id = self._find_folder_by_name(folder_name, parent_id=syftbox_folder_id)
        if not folder_id:
            folder_id = self.create_folder(folder_name, syftbox_folder_id)
        self._encryption_bundles_folder_id = folder_id
        return folder_id

    def _encryption_bundle_filename(self, owner_email: str, peer_email: str) -> str:
        return f"encryption_bundle_{owner_email}_for_{peer_email}.json"

    def write_encryption_bundle(self, peer_email: str, bundle_json: str) -> None:
        """Write own encryption bundle for a peer to own bundles folder."""
        folder_id = self._get_or_create_encryption_bundles_folder_id()
        filename = self._encryption_bundle_filename(self.email, peer_email)
        file_payload, _ = self.create_file_payload(bundle_json)

        # Check if file already exists
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])

        if items:
            execute_with_retries(
                self.drive_service.files().update(
                    fileId=items[0]["id"], media_body=file_payload
                )
            )
        else:
            file_metadata = {"name": filename, "parents": [folder_id]}
            execute_with_retries(
                self.drive_service.files().create(
                    body=file_metadata,
                    media_body=file_payload,
                    fields="id",
                )
            )

    def share_encryption_bundles_folder(self, peer_email: str) -> None:
        """Share the bundles folder with a peer so they can read bundles."""
        folder_id = self._get_or_create_encryption_bundles_folder_id()
        self.add_permission(folder_id, peer_email, write=False)

    def read_peer_encryption_bundle(self, peer_email: str) -> str | None:
        """Read encryption bundle that peer wrote for us.

        Searches for: encryption_bundle_{peer_email}_for_{self.email}.json
        in peer's bundles folder.
        """
        filename = self._encryption_bundle_filename(peer_email, self.email)
        query = f"name='{filename}' and trashed=false and '{peer_email}' in owners"
        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        if not items:
            return None
        try:
            data = self.download_file(items[0]["id"])
            return data.decode("utf-8")
        except Exception:
            return None
