"""Google Drive Files transport layer implementation"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import build_http
from pydantic import BaseModel
from syft_migration import MigrationError

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
from syft.sync.connections.base_connection import (
    FileCollection,
    SyftboxPlatformConnection,
)
from syft.sync.connections.drive.drive_connections import drive_transport
from syft.sync.connections.drive.gdrive_client import (
    GOOGLE_FOLDER_MIME_TYPE,
    SYFT_FILE_PROPERTY_KEY,
    SYFT_FILE_PROPERTY_VALUE,
    GDriveClient,
)
from syft.sync.connections.drive.gdrive_layout import (
    GDRIVE_P2P_FOLDER_DATASITE_PREFIX,
    SYFTBOX_FOLDER,
    DriveLayout,
    GdriveP2PFolder,
)
from syft.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
    raise_unavailable,
)
from syft.sync.connections.drive.gdrive_utils import (
    gather_all_file_and_folder_ids_recursive,
)
from syft.sync.environments.environment import Environment
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)
from syft.sync.messages.proposed_filechange import (
    FileNameParseError,
    MessageFileName,
    ProposedFileChangesMessage,
)
from syft.sync.transport.naming import (
    PEERS_META_KEY,
    SYFT_PEERS_VERSION,
    timestamp_from_message_name,
)
from syft.sync.utils.syftbox_utils import check_env
from syft.version import SYFT_VERSION

if TYPE_CHECKING:
    from syft.sync.connections.drive.grdrive_config import (
        GdriveConnectionConfig,
    )
    from syft.sync.version.peer_info import PeerInfo

# Timeout for Google API requests (in seconds)
GOOGLE_API_TIMEOUT = 120  # 2 minutes

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
        import google.auth
        from google.colab import auth as colab_auth

        colab_auth.authenticate_user()
        creds, _ = google.auth.default()
        authed_http = AuthorizedHttp(creds, http=http)
        # Build service without explicit credentials in Colab
        return build("drive", "v3", http=authed_http)
    else:
        authorized_http = AuthorizedHttp(credentials, http=http)
        return build("drive", "v3", http=authorized_http)


SYFT_PEERS_FILE = "SYFT_peers.json"

SYFT_VERSION_FILE = "SYFT_version.json"


class GdriveEncryptionBundlesFolder(BaseModel):
    """Folder for encryption bundles: syft_encryption_bundles#email"""

    email: str

    def as_string(self) -> str:
        return f"syft_encryption_bundles#{self.email}"


def collection_name_query(prefix: str) -> str:
    """A Drive query that finds a collection of ``prefix`` in any layout.

    It has no trailing '_', because a layout writes its variant in that
    position. A client that predates multi-layout searches with the '_', and so
    never lists a layout it cannot read.
    """
    return f"name contains '{prefix}'"


@lru_cache(maxsize=None)
def _collection_name_re(prefix: str) -> "re.Pattern[str]":
    """Matches '{prefix}{variant}_{tag}_{hash}'.

    The variant and the hash hold no '_'. The tag can hold one, so it takes
    every character in between.
    """
    return re.compile(
        rf"^{re.escape(prefix)}"
        r"(?P<variant>[^_]*)_(?P<tag>.+)_(?P<content_hash>[^_]+)$"
    )


class CollectionFolder(BaseModel):
    """Naming value object for collection folders (domain-agnostic).

    Collections (datasets, private datasets, or any future collection type) are
    named ``{prefix}{variant}_{tag}_{content_hash}`` on the wire. The variant
    names the layout (see CollectionSyncSpec) and is "" for the original one, so
    that name is unchanged byte for byte. This is the single source of truth for
    building and parsing the name; it also computes the content hash from the
    folder's file contents.

    A writer normally folds the variant into ``prefix`` (the spec builds the
    qualified prefix once) and leaves ``variant`` empty; ``from_name`` splits
    them apart again for a reader.
    """

    prefix: str
    tag: str
    content_hash: str
    variant: str = ""

    @property
    def folder_name(self) -> str:
        """The on-wire folder name. Doubles as the cache key (same string)."""
        return f"{self.prefix}{self.variant}_{self.tag}_{self.content_hash}"

    @classmethod
    def from_name(cls, prefix: str, name: str) -> "CollectionFolder":
        """Parse '{prefix}{variant}_{tag}_{hash}' into a CollectionFolder.

        Raises ValueError if the name does not carry the prefix, so callers can
        skip non-matching folders.
        """
        match = _collection_name_re(prefix).match(name)
        if match is None:
            raise ValueError(f"Invalid collection folder name: {name}")
        return cls(prefix=prefix, **match.groupdict())

    @staticmethod
    def compute_hash(files: dict[str, bytes]) -> str:
        """Compute a content hash from file contents."""
        from syft.sync.file_utils import compute_file_hashes

        return compute_file_hashes(files)


class GDriveConnection(SyftboxPlatformConnection):
    """Google Drive Files API transport layer"""

    class Config:
        arbitrary_types_allowed = True

    layout: DriveLayout | None = None
    credentials: GoogleCredentials | None = None
    verbose: bool = True
    email: str
    token_path: Path | None = None
    _is_setup: bool = False

    # fname -> gdrive id
    personal_syftbox_event_id_cache: dict[str, str] = {}

    # tag -> dataset collection folder id
    collection_folder_id_cache: Dict[str, str] = {}

    # Rolling state caches for single-API-call optimization
    _rolling_state_folder_id: str | None = None
    _rolling_state_file_id: str | None = None

    # Encryption bundles folder cache
    _encryption_bundles_folder_id: str | None = None

    # Cached SYFT_peers.json contents (None = not loaded yet).
    _peers_json_cache: dict[str, dict[str, Any]] | None = None

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
        from syft.sync.connections.drive.mock_drive_service import (
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
        if drive_service is None:
            drive_service = build_drive_service(
                self.credentials, environment=self.environment
            )
        self.layout = DriveLayout(GDriveClient(drive_service), self.email)

        if warm_syftbox_folder_id_cache:
            self.layout.get_personal_syftbox_folder_id()
        self._is_setup = True

    @property
    def client(self) -> GDriveClient:
        return self.layout.client

    def as_transport(self):
        """This connection as the transport jobs. See ``drive_connections.py``."""
        return drive_transport(self)

    def clone_for_thread(self) -> "GDriveConnection":
        """A connection safe to use from another thread: the Drive client is not."""
        return self.copy()

    def copy(self) -> "GDriveConnection":
        # if is mock
        from syft.sync.connections.drive.mock_drive_service import (
            MockDriveService,
        )

        if isinstance(self.client.drive_service, MockDriveService):
            new_conn = GDriveConnection.from_service(
                self.email, self.client.drive_service
            )
        else:
            new_conn = GDriveConnection.from_token_path(
                self.email, self.token_path, warm_syftbox_folder_id_cache=False
            )
        self._copy_caches_to(new_conn)
        return new_conn

    def _copy_caches_to(self, other: "GDriveConnection") -> None:
        """Seed all in-memory caches on `other` from `self`, so a freshly-built
        copy doesn't re-discover folder ids via Drive search on first use."""
        self.layout.copy_caches_to(other.layout)
        other._rolling_state_folder_id = self._rolling_state_folder_id
        other._rolling_state_file_id = self._rolling_state_file_id
        other._encryption_bundles_folder_id = self._encryption_bundles_folder_id
        other._peers_json_cache = (
            dict(self._peers_json_cache) if self._peers_json_cache is not None else None
        )
        other.personal_syftbox_event_id_cache = dict(
            self.personal_syftbox_event_id_cache
        )
        other.collection_folder_id_cache = dict(self.collection_folder_id_cache)

    @property
    def environment(self) -> Environment:
        return check_env()

    def get_authenticated_email(self) -> str:
        """Return the email of the Google account behind the Drive service."""
        about = execute_with_retries(
            self.client.drive_service.about().get(fields="user(emailAddress)")
        )
        return about["user"]["emailAddress"]

    def create_peer_datasite_folders(self, peer_email: str):
        """Create inbox + outbox folders for peer's datasite on own drive, share with peer.

        Creates:
        - syft_datasite_{peer}_inbox_{self} — I submit proposals to peer here
        - syft_datasite_{peer}_outbox_{self} — peer pushes events to me here
        """
        # Inbox: I write proposals, peer reads
        inbox_id = self.layout.get_peer_datasite_inbox_id(peer_email)
        if inbox_id is None:
            inbox_id = self.layout.create_peer_datasite_folder(peer_email, "inbox")
        self.client.add_permission(inbox_id, peer_email, write=True)

        # Outbox: peer writes events, I read
        outbox_id = self.layout.get_peer_datasite_outbox_id(peer_email)
        if outbox_id is None:
            outbox_id = self.layout.create_peer_datasite_folder(peer_email, "outbox")
        self.client.add_permission(outbox_id, peer_email, write=True)

    def add_peer(self, peer_email: str):
        """Alias for create_peer_datasite_folders."""
        self.create_peer_datasite_folders(peer_email)

    def _get_peers_file_id(self) -> str | None:
        """Find SYFT_peers.json file in /SyftBox folder"""
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        query = f"name='{SYFT_PEERS_FILE}' and '{syftbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def _download_peers_json(self) -> dict[str, dict[str, Any]]:
        """Fetch peers JSON from GDrive. Returns empty dict if not found.

        Maps each peer email to ``{"state": str, "public_encryption_bundle":
        dict}``, and ``_meta`` to ``{"version": int}``.

        Raises:
            TransportUnavailableError: when Drive cannot be reached.
            ValueError: when the file is not valid JSON.
        """
        with raise_unavailable("Could not read the peers file from Drive."):
            file_id = self._get_peers_file_id()
            if file_id is None:
                return {}
            file_data = self.client.download_file(file_id)
        try:
            return json.loads(file_data.decode("utf-8"))
        except ValueError as e:
            raise ValueError(f"Could not read the peers file: {e}") from e

    def _get_peers_json(
        self, force_download: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Return peers JSON, using the in-memory cache when available."""
        if self._peers_json_cache is not None and not force_download:
            return self._peers_json_cache
        self._peers_json_cache = self._download_peers_json()
        return self._peers_json_cache

    def _write_peers_json(self, peers_data: dict[str, dict[str, Any]]):
        """Write peers JSON to GDrive. Creates or updates the file."""
        peers_data = {
            **peers_data,
            PEERS_META_KEY: {"version": SYFT_PEERS_VERSION},
        }
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        file_id = self._get_peers_file_id()

        # Convert to JSON bytes
        json_data = json.dumps(peers_data, indent=2)
        file_payload, _ = self.client.create_file_payload(json_data)

        if file_id is None:
            # Create new file
            file_metadata = {
                "name": SYFT_PEERS_FILE,
                "parents": [syftbox_folder_id],
            }
            result = self.client.create_syft_file(file_metadata, file_payload)
            self._peers_json_cache = peers_data
            return result.get("id")
        else:
            # Update existing file
            execute_with_retries(
                self.client.drive_service.files().update(
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

    def get_peer_requests(self) -> list[str]:
        """Get list of pending peer requests.

        Scans for syft_datasite_#version#{self}_*  folders NOT owned by self — those are
        peers who created folders for our datasite. Filters out already-accepted
        or rejected peers from SYFT_peers.json.
        """
        folders = self.client.list_all_files(
            f"name contains '{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#' "
            f"and name contains '#{self.email}#' "
            f"and trashed=false "
            f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
            f"and not 'me' in owners",
            fields="id,name,owners",
        )

        all_folder_peers = set()
        for f in folders:
            try:
                folder = GdriveP2PFolder.from_name(f["name"])
            except ValueError:
                # The query matches a name prefix, so a folder with another shape
                # can appear here.
                continue
            # Any account can make a folder with this name. The request is from
            # the owner, so the name must name the owner.
            owners = f.get("owners", [])
            if (
                len(owners) != 1
                or owners[0].get("emailAddress", "").casefold()
                != folder.peer_email.casefold()
            ):
                continue
            if folder.datasite_email == self.email:
                all_folder_peers.add(folder.peer_email)

        peers_data = self._get_peers_json()
        pending_peers = []
        for peer_email in all_folder_peers:
            undecided = peers_data.get(peer_email, {}).get("state") not in [
                "accepted",
                "rejected",
            ]
            if undecided:
                pending_peers.append(peer_email)

        return pending_peers

    def watcher_download_raw_events_from_outbox(
        self, peer_email: str, since_timestamp: float | None
    ) -> list[bytes]:
        folder_id = self.layout.get_peer_datasite_outbox_id(peer_email)
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
                res.append(self.client.download_file(name_to_id[file_name]))
        return res

    def watcher_get_events_messages(
        self, peer_email: str, since_timestamp: float | None
    ) -> list[FileChangeEventsMessage]:
        raw_list = self.watcher_download_raw_events_from_outbox(
            peer_email, since_timestamp
        )
        return [FileChangeEventsMessage.from_compressed_data(data) for data in raw_list]

    def watcher_get_outbox_file_metadatas(
        self, peer_email: str, since_timestamp: float | None
    ) -> list[dict]:
        """Get file metadata from peer's outbox folder without downloading."""
        folder_id = self.layout.get_peer_datasite_outbox_id(peer_email)
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
        personal_syftbox_folder_id = self.layout.get_personal_syftbox_folder_id()
        file_metadata = {
            "name": filename,
            "parents": [personal_syftbox_folder_id],
        }
        file_payload, _ = self.client.create_file_payload(data)

        res = self.client.create_syft_file(file_metadata, file_payload)
        gdrive_id = res.get("id")
        self.personal_syftbox_event_id_cache[filename] = gdrive_id
        return gdrive_id

    def owner_download_raw_bytes_by_id(self, file_id: str) -> bytes:
        """Download raw bytes by file ID."""
        return self.client.download_file(file_id)

    def owner_get_all_accepted_event_file_ids(
        self, since_timestamp: float | None = None
    ) -> list[str]:
        personal_syftbox_folder_id = self.layout.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(
            personal_syftbox_folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = self._filter_valid_file_metadatas(file_metadatas)
        oldest_first = sorted(
            valid_fname_objs,
            key=lambda f: (
                FileChangeEventsMessageFileName.from_string(f["name"]).timestamp
            ),
        )
        return [f["id"] for f in oldest_first]

    @raise_unavailable("Could not read the event log from Drive.")
    def owner_download_all_raw_events_from_syftbox(self) -> list[bytes]:
        """Download all event files from /SyftBox/myemail as raw bytes."""
        personal_syftbox_folder_id = self.layout.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(personal_syftbox_folder_id)
        valid_fname_objs = self._get_valid_events_from_file_metadatas(file_metadatas)

        result = []
        for fname_obj in sorted(valid_fname_objs, key=lambda x: x.timestamp):
            gdrive_id = [
                f for f in file_metadatas if f["name"] == fname_obj.as_string()
            ][0]["id"]
            result.append(self.client.download_file(gdrive_id))
        return result

    def owner_write_raw_bytes_to_outbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        outbox_folder_id = self.layout.get_own_datasite_outbox_id(recipient)
        if outbox_folder_id is None:
            raise ValueError(f"Outbox folder for {recipient} not found")

        file_payload, _ = self.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [outbox_folder_id]}

        result = self.client.create_syft_file(
            file_metadata, file_payload, fields="id, parents"
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
                self.client.drive_service.files().update(
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
        self,
        proposed_filechange_message: ProposedFileChangesMessage,
        sender_email: str,
    ):
        """Archive a handled message. ``sender_email`` is the peer it came from.

        The caller reads that email from the transport. The ``sender_email``
        field of the message is sender-supplied, and it names the archive folder,
        so this method never reads it.
        """
        fname = proposed_filechange_message.message_filename.as_string()

        # Use cached platform_id if available, otherwise fall back to name-based lookup
        gdrive_id = proposed_filechange_message.platform_id
        if gdrive_id is None:
            gdrive_id = self.get_inbox_proposed_event_id_from_name(sender_email, fname)
        if gdrive_id is None:
            raise ValueError(
                f"Event {fname} not found in inbox, event should already be created for this type of connection"
            )
        self.owner_archive_inbox_file(gdrive_id, sender_email)

    def owner_archive_inbox_file(self, file_id: str, sender_email: str) -> None:
        """Move the inbox file ``file_id`` from ``sender_email`` to its archive folder."""
        file_info = execute_with_retries(
            self.client.drive_service.files().get(fileId=file_id, fields="parents")
        )
        previous_parents = ",".join(file_info.get("parents", []))
        archive_folder_id = self.layout.get_archive_folder_id(sender_email)
        execute_with_retries(
            self.client.drive_service.files().update(
                fileId=file_id,
                addParents=archive_folder_id,
                removeParents=previous_parents,
                fields="id, parents",
                supportsAllDrives=True,
            )
        )

    def get_file_metadatas_from_folder(
        self,
        folder_id: str,
        since_timestamp: float | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        """
        Get file metadatas from folder with early termination.

        Args:
            folder_id: Google Drive folder ID
            since_timestamp: Optional timestamp. If provided, only files with a
                           newer timestamp are returned, and pagination stops at
                           a file from an earlier second.
            page_size: Number of files to fetch per API call. Default 100.

        Returns:
            List of file metadata dicts, sorted by name descending (newest first)
        """
        query = f"'{folder_id}' in parents and trashed=false"
        all_files = []
        page_token = None

        while True:
            results = execute_with_retries(
                self.client.drive_service.files().list(
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
                    timestamp = timestamp_from_message_name(file["name"])

                    if timestamp is not None:
                        if timestamp > since_timestamp:
                            all_files.append(file)
                        elif int(timestamp) < int(since_timestamp):
                            # Drive sorts names as strings, so a 0.9 fraction of
                            # a second lists before 0.95. We stop when the
                            # second flips.
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

    @staticmethod
    def _get_valid_events_from_file_metadatas(
        file_metadatas: list[dict],
    ) -> list[FileChangeEventsMessageFileName]:
        res = []
        for file_metadata in file_metadatas:
            fname = file_metadata["name"]
            try:
                message_filename = FileChangeEventsMessageFileName.from_string(fname)
            except ValueError:
                print(f"Warning: invalid event file name: {fname}")
                continue
            res.append(message_filename)
        return res

    @staticmethod
    def _get_valid_messages_from_file_metadatas(
        file_metadatas: list[dict],
    ) -> list[MessageFileName]:
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
        inbox_folder_id = self.layout.get_own_datasite_inbox_id(sender_email)
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
        raw_data = self.client.download_file(first_file_id)
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

    def owner_list_outbox_files(self, recipient_email: str) -> list[dict] | None:
        """The files in my outbox for ``recipient_email``, or None when that
        outbox does not exist."""
        folder_id = self.layout.get_own_datasite_outbox_id(recipient_email)
        if folder_id is None:
            return None
        return self.get_file_metadatas_from_folder(folder_id, since_timestamp=None)

    def watcher_send_raw_bytes_to_inbox(
        self, recipient: str, filename: str, data: bytes
    ) -> None:
        inbox_id = self.layout.get_peer_datasite_inbox_id(recipient)
        if inbox_id is None:
            raise Exception(f"Inbox folder for {recipient}'s datasite not found")

        payload, _ = self.client.create_file_payload(data)
        file_metadata = {"name": filename, "parents": [inbox_id]}

        result = self.client.create_syft_file(
            file_metadata, payload, fields="id, parents"
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
                self.client.drive_service.files().update(
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
        self.layout.reset_caches()
        self.personal_syftbox_event_id_cache.clear()
        self.collection_folder_id_cache.clear()
        self._rolling_state_folder_id = None
        self._rolling_state_file_id = None
        self._encryption_bundles_folder_id = None
        self._peers_json_cache = None

    def gather_all_file_and_folder_ids(self) -> list[str]:
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        return gather_all_file_and_folder_ids_recursive(
            self.client.drive_service, syftbox_folder_id
        )

    def download_file(self, file_id: str) -> bytes:
        return self.client.download_file(file_id)

    def delete_file_by_id(self, file_id: str, raise_on_error: bool = False):
        self.client.delete_file_by_id(file_id, raise_on_error=raise_on_error)

    def delete_multiple_files_by_ids(
        self,
        file_ids: list[str],
        ignore_permissions_errors: bool = True,
        ignore_file_not_found: bool = True,
    ):
        self.client.delete_multiple_files_by_ids(
            file_ids,
            ignore_permissions_errors=ignore_permissions_errors,
            ignore_file_not_found=ignore_file_not_found,
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
        return [f["id"] for f in self.client.list_all_files(query, fields="id")]

    def get_inbox_proposed_event_id_from_name(
        self, sender_email: str, name: str
    ) -> str | None:
        inbox_folder_id = self.layout.get_own_datasite_inbox_id(sender_email)
        query = f"name='{name}' and '{inbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    # =========================================================================
    # GENERIC COLLECTION METHODS (parameterized on `prefix`)
    #
    # Folder names are built and parsed via CollectionFolder, the single source
    # of truth for the f"{prefix}_{tag}_{content_hash}" scheme. On-wire behavior
    # is byte-identical to the previous inline construction.
    # =========================================================================

    def owner_create_collection_folder(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> str:
        """Create /SyftBox/{prefix}_{tag}_{hash} folder."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name

        # Check cache
        if folder_name in self.collection_folder_id_cache:
            return self.collection_folder_id_cache[folder_name]

        syftbox_folder_id = self.layout.get_syftbox_folder_id()

        # Check if exists
        folder_id = self.client.find_folder_by_name(
            folder_name, parent_id=syftbox_folder_id
        )
        if folder_id:
            self.collection_folder_id_cache[folder_name] = folder_id
            return folder_id

        # Create new folder
        folder_id = self.client.create_folder(folder_name, syftbox_folder_id)
        self.collection_folder_id_cache[folder_name] = folder_id
        return folder_id

    def owner_tag_collection_as_any(
        self, prefix: str, tag: str, content_hash: str
    ) -> None:
        """Mark collection as shared with 'any' via appProperties."""
        folder_id = self._get_collection_folder_id(prefix, tag, content_hash)
        execute_with_retries(
            self.client.drive_service.files().update(
                fileId=folder_id,
                body={"appProperties": {"syft_shared_with_any": "true"}},
            )
        )

    def owner_share_collection(
        self, prefix: str, tag: str, content_hash: str, users: list[str]
    ) -> None:
        """Share collection folder with specific users via batch API."""
        if not users:
            return
        folder_id = self._get_collection_folder_id(prefix, tag, content_hash)
        self.client.batch_add_permissions(folder_id, users)

    def owner_upload_collection_files(
        self, prefix: str, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        """Upload files to a collection folder."""
        folder_id = self._get_collection_folder_id(prefix, tag, content_hash)

        for file_path, content in files.items():
            file_payload, _ = self.client.create_file_payload(content)
            file_name = Path(file_path).name

            file_metadata = {"name": file_name, "parents": [folder_id]}
            self.client.create_syft_file(file_metadata, file_payload)

    def owner_list_collections(self, prefix: str) -> list[str]:
        """List collections created by DO (owned by me). Returns list of tags.

        A tag appears once however many layouts hold it.
        """
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        query = (
            f"{collection_name_query(prefix)} and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        result: list[str] = []
        for folder in self.client.list_all_files(query, fields="name"):
            try:
                cf = CollectionFolder.from_name(prefix, folder["name"])
            except ValueError:
                continue
            if cf.tag not in result:
                result.append(cf.tag)
        return result

    def owner_list_all_collections_with_permissions(
        self,
        prefix: str,
    ) -> list[FileCollection]:
        """List all DO's collections with permissions info, one per layout."""
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        query = (
            f"{collection_name_query(prefix)} and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        collections = []
        for folder in self.client.list_all_files(query, fields="id,name,appProperties"):
            try:
                cf = CollectionFolder.from_name(prefix, folder["name"])
            except ValueError:
                continue
            has_anyone = (
                folder.get("appProperties", {}).get("syft_shared_with_any") == "true"
            )
            collections.append(
                FileCollection(
                    folder_id=folder["id"],
                    tag=cf.tag,
                    content_hash=cf.content_hash,
                    has_any_permission=has_anyone,
                    variant=cf.variant,
                )
            )

        return collections

    def owner_delete_collection(self, prefix: str, tag: str) -> None:
        """Delete every layout of ``tag`` published under ``prefix``."""
        collections = self.owner_list_all_collections_with_permissions(prefix)
        for c in collections:
            if c.tag == tag:
                self.client.delete_file_by_id(c.folder_id)
                folder_name = CollectionFolder(
                    prefix=prefix,
                    tag=c.tag,
                    content_hash=c.content_hash,
                    variant=c.variant,
                ).folder_name
                self.collection_folder_id_cache.pop(folder_name, None)

    def watcher_list_collections(self, prefix: str) -> list[dict]:
        """List collections shared with DS (not owned by me), one per layout.

        Returns list of dicts with keys: owner_email, tag, content_hash, variant
        """
        query = (
            f"{collection_name_query(prefix)} and not 'me' in owners "
            f"and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        result = []
        for folder in self.client.list_all_files(query, fields="name, owners"):
            try:
                cf = CollectionFolder.from_name(prefix, folder["name"])
            except ValueError:
                # Skip folders that don't match the expected format
                continue
            owner_email = folder.get("owners", [{}])[0].get("emailAddress", "unknown")
            result.append(
                {
                    "owner_email": owner_email,
                    "tag": cf.tag,
                    "content_hash": cf.content_hash,
                    "variant": cf.variant,
                }
            )
        return result

    def watcher_download_collection(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        """Download all files from a collection."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name
        # Try to find folder by name (could be owned by someone else)
        folder_id = self.client.find_folder_by_name(
            folder_name, owner_email=owner_email
        )

        if not folder_id:
            raise ValueError(f"Collection {tag} with hash {content_hash} not found")

        file_metadatas = self.get_file_metadatas_from_folder(folder_id)
        files = {}
        for file_meta in file_metadatas:
            file_id = file_meta["id"]
            file_name = file_meta["name"]
            files[file_name] = self.client.download_file(file_id)

        return files

    def watcher_get_collection_file_metadatas(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> List[Dict]:
        """Get file metadata from a collection without downloading."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name
        folder_id = self.client.find_folder_by_name(
            folder_name, owner_email=owner_email
        )

        if not folder_id:
            raise ValueError(f"Collection {tag} with hash {content_hash} not found")

        file_metadatas = self.get_file_metadatas_from_folder(folder_id)
        return [{"file_id": f["id"], "file_name": f["name"]} for f in file_metadatas]

    def watcher_download_collection_file(self, file_id: str) -> bytes:
        """Download a single file from a collection."""
        return self.client.download_file(file_id)

    def _get_collection_folder_id(
        self, prefix: str, tag: str, content_hash: str
    ) -> str:
        """Get folder ID for a collection, with caching."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name
        if folder_name in self.collection_folder_id_cache:
            return self.collection_folder_id_cache[folder_name]

        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        folder_id = self.client.find_folder_by_name(
            folder_name, parent_id=syftbox_folder_id
        )

        if not folder_id:
            raise ValueError(
                f"Collection folder {tag} with hash {content_hash} not found"
            )

        self.collection_folder_id_cache[folder_name] = folder_id
        return folder_id

    def _get_own_peer_info_file_id(self) -> Optional[str]:
        """Find the own peer info file, SYFT_version.json, in /SyftBox."""
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        query = f"name='{SYFT_VERSION_FILE}' and '{syftbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def write_peer_info_file(self, peer_info: "PeerInfo") -> None:
        """Write the peer info file to /SyftBox. Creates or updates the file."""

        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        file_id = self._get_own_peer_info_file_id()

        # Convert to JSON string
        json_data = peer_info.to_json()
        file_payload, _ = self.client.create_file_payload(json_data)

        if file_id is None:
            # Create new file
            file_metadata = {
                "name": SYFT_VERSION_FILE,
                "parents": [syftbox_folder_id],
            }
            self.client.create_syft_file(file_metadata, file_payload)
        else:
            # Update existing file
            execute_with_retries(
                self.client.drive_service.files().update(
                    fileId=file_id, media_body=file_payload
                )
            )

    def _get_peer_info_file_id_of(self, peer_email: str) -> Optional[str]:
        """Find the peer info file, SYFT_version.json, of ``peer_email``."""
        query = (
            f"name='{SYFT_VERSION_FILE}' and trashed=false and '{peer_email}' in owners"
        )
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def read_own_peer_info_file(self) -> Optional["PeerInfo"]:
        """Read the own peer info file from /SyftBox.

        Returns None when there is no file.

        Raises:
            TransportUnavailableError: when Drive cannot be reached.
            ValueError: when the file is not a peer info file this client reads.
        """
        return self._read_peer_info_file(
            self._get_own_peer_info_file_id, "the own peer info file"
        )

    def read_peer_info_file(self, peer_email: str) -> Optional["PeerInfo"]:
        """Read the peer info file from a peer's /SyftBox folder.

        Returns None when the peer has no file that this account may read.

        Raises:
            TransportUnavailableError: when Drive cannot be reached.
            ValueError: when the file is not a peer info file this client reads.
        """
        return self._read_peer_info_file(
            lambda: self._get_peer_info_file_id_of(peer_email),
            f"the peer info file of {peer_email}",
        )

    def _read_peer_info_file(
        self, find_file_id: Callable[[], str | None], file_label: str
    ) -> Optional["PeerInfo"]:
        """Find, download and parse a peer info file.

        Returns None when ``find_file_id`` finds no file. ``file_label`` names
        the file in errors.
        """
        from syft.sync.version.peer_info import PeerInfo

        with raise_unavailable(f"Could not read {file_label} from Drive."):
            file_id = find_file_id()
            if file_id is None:
                return None
            file_data = self.client.download_file(file_id)
        try:
            return PeerInfo.from_json(file_data.decode("utf-8"))
        except (ValueError, MigrationError) as e:
            raise ValueError(f"Could not read {file_label}: {e}") from e

    def share_peer_info_file_with_peer(
        self, peer_email: str, peer_info: "PeerInfo"
    ) -> None:
        """Share the peer info file with a peer so they can read it.

        A missing file is created from ``peer_info``, the info of this client.
        """
        file_id = self._get_own_peer_info_file_id()
        if file_id is None:
            self.write_peer_info_file(peer_info)
            file_id = self._get_own_peer_info_file_id()

        if file_id:
            self.client.add_permission(file_id, peer_email, write=False)

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def _get_checkpoints_folder_name(self) -> str:
        """Get the checkpoints folder name: {email}-{version}-checkpoints"""
        return f"{self.email}-{SYFT_VERSION}-checkpoints"

    def _get_checkpoints_folder_id(self) -> str | None:
        """Find the checkpoints folder ID from Google Drive."""
        folders = self.client.find_folders(
            name_contains=[f"{self.email}-", "-checkpoints"],
            parent_id=self.layout.get_syftbox_folder_id(),
        )
        return self.layout.find_or_adopt_versioned_folder(
            folders, current_name=self._get_checkpoints_folder_name()
        )

    def _get_or_create_checkpoints_folder_id(self) -> str:
        """Get or create the checkpoints folder."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is not None:
            return folder_id
        # Create the folder
        folder_name = self._get_checkpoints_folder_name()
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        return self.client.create_folder(folder_name, syftbox_folder_id)

    def upload_raw_checkpoint(self, filename: str, data: bytes) -> str:
        """Upload raw checkpoint bytes to Google Drive.

        Uploads the new checkpoint first, then deletes old ones.
        """
        folder_id = self._get_or_create_checkpoints_folder_id()
        payload, _ = self.client.create_file_payload(data)

        file_metadata = {
            "name": filename,
            "parents": [folder_id],
        }

        result = self.client.create_syft_file(file_metadata, payload)

        # Only delete old checkpoints after successful upload
        self.delete_all_checkpoints(exclude_file_id=result.get("id"))

        return result.get("id")

    @raise_unavailable("Could not read the latest checkpoint from Drive.")
    def download_raw_latest_checkpoint(self) -> bytes | None:
        """Download the latest checkpoint as raw bytes, or None."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return None

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{CHECKPOINT_FILENAME_PREFIX}'"
        )
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
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

        return self.client.download_file(latest_file["id"])

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
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
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
                    self.client.drive_service.files().delete(fileId=item["id"])
                )
            except Exception as e:
                print(f"Warning: Failed to delete checkpoint {item['name']}: {e}")

    # =========================================================================
    # INCREMENTAL CHECKPOINT METHODS
    # =========================================================================

    def upload_raw_incremental_checkpoint(self, filename: str, data: bytes) -> str:
        """Upload raw incremental checkpoint bytes to Google Drive."""
        folder_id = self._get_or_create_checkpoints_folder_id()
        payload, _ = self.client.create_file_payload(data)

        file_metadata = {
            "name": filename,
            "parents": [folder_id],
        }

        result = self.client.create_syft_file(file_metadata, payload)
        return result.get("id")

    @raise_unavailable("Could not read the incremental checkpoints from Drive.")
    def download_all_raw_incremental_checkpoints(self) -> list[bytes]:
        """Download all incremental checkpoints as raw bytes."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return []

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])

        if not items:
            return []

        return [self.client.download_file(item["id"]) for item in items]

    def get_incremental_checkpoint_count(self) -> int:
        """Get the number of incremental checkpoints on Google Drive."""
        folder_id = self._get_checkpoints_folder_id()
        if folder_id is None:
            return 0

        query = (
            f"'{folder_id}' in parents and trashed=false "
            f"and name contains '{INCREMENTAL_CHECKPOINT_PREFIX}'"
        )
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
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
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(name)")
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
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])

        for item in items:
            try:
                execute_with_retries(
                    self.client.drive_service.files().delete(fileId=item["id"])
                )
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
        personal_folder_id = self.layout.get_personal_syftbox_folder_id()
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
            event_timestamp = timestamp_from_message_name(f["name"])
            if event_timestamp is not None and event_timestamp > checkpoint_timestamp:
                count += 1
        return count

    @raise_unavailable("Could not read the event log from Drive.")
    def download_raw_events_since_timestamp(
        self, since_timestamp: float
    ) -> list[bytes]:
        """Download event files created after a timestamp as raw bytes."""
        personal_folder_id = self.layout.get_personal_syftbox_folder_id()
        file_metadatas = self.get_file_metadatas_from_folder(
            personal_folder_id, since_timestamp=since_timestamp
        )

        valid_fname_objs = self._get_valid_events_from_file_metadatas(file_metadatas)

        result = []
        for fname_obj in sorted(valid_fname_objs, key=lambda x: x.timestamp):
            if fname_obj.timestamp <= since_timestamp:
                continue

            gdrive_id = [
                f for f in file_metadatas if f["name"] == fname_obj.as_string()
            ][0]["id"]

            result.append(self.client.download_file(gdrive_id))

        return result

    # =========================================================================
    # ROLLING STATE METHODS
    # =========================================================================

    def _get_rolling_state_folder_name(self) -> str:
        """Get the rolling state folder name: {email}-{version}-rolling-state"""
        return f"{self.email}-{SYFT_VERSION}-rolling-state"

    def _get_rolling_state_folder_id(self, use_cache: bool = True) -> str | None:
        """
        Find the rolling state folder ID from Google Drive.

        Args:
            use_cache: If True, return cached value if available.
        """
        if use_cache and self._rolling_state_folder_id is not None:
            return self._rolling_state_folder_id

        folders = self.client.find_folders(
            name_contains=[f"{self.email}-", "-rolling-state"],
            parent_id=self.layout.get_syftbox_folder_id(),
        )
        folder_id = self.layout.find_or_adopt_versioned_folder(
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
        syftbox_folder_id = self.layout.get_syftbox_folder_id()
        folder_id = self.client.create_folder(folder_name, syftbox_folder_id)
        self._rolling_state_folder_id = folder_id
        return folder_id

    def upload_raw_rolling_state(self, filename: str, data: bytes) -> str:
        """Upload raw rolling state bytes to Google Drive.

        Optimized to use update() if file ID is cached.
        """
        payload, _ = self.client.create_file_payload(data)

        # Try to update existing file if we have a cached ID
        if self._rolling_state_file_id is not None:
            try:
                execute_with_retries(
                    self.client.drive_service.files().update(
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
            "name": filename,
            "parents": [folder_id],
        }

        result = self.client.create_syft_file(file_metadata, payload)
        file_id = result.get("id")
        self._rolling_state_file_id = file_id
        return file_id

    @raise_unavailable("Could not read the rolling state from Drive.")
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
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
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

        data = self.client.download_file(latest_file["id"])
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
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])

        # Delete each rolling state file
        for item in items:
            try:
                execute_with_retries(
                    self.client.drive_service.files().delete(fileId=item["id"])
                )
            except Exception as e:
                print(f"Warning: Failed to delete rolling state {item['name']}: {e}")

    # =========================================================================
    # ENCRYPTION BUNDLE METHODS
    # =========================================================================

    def _find_encryption_bundles_folder_id(self) -> str | None:
        """The id of the encryption bundles folder of this user, or None when it does not exist."""
        if self._encryption_bundles_folder_id:
            return self._encryption_bundles_folder_id
        folder_name = GdriveEncryptionBundlesFolder(email=self.email).as_string()
        folder_id = self.client.find_folder_by_name(
            folder_name, parent_id=self.layout.get_syftbox_folder_id()
        )
        if folder_id:
            self._encryption_bundles_folder_id = folder_id
        return folder_id

    def _get_or_create_encryption_bundles_folder_id(self) -> str:
        """Get or create the encryption bundles folder for this user."""
        folder_id = self._find_encryption_bundles_folder_id()
        if not folder_id:
            folder_name = GdriveEncryptionBundlesFolder(email=self.email).as_string()
            folder_id = self.client.create_folder(
                folder_name, self.layout.get_syftbox_folder_id()
            )
            self._encryption_bundles_folder_id = folder_id
        return folder_id

    def _encryption_bundle_filename(self, owner_email: str, peer_email: str) -> str:
        return f"encryption_bundle_{owner_email}_for_{peer_email}.json"

    def write_encryption_bundle(self, peer_email: str, bundle_json: str) -> None:
        """Write own encryption bundle for a peer to own bundles folder."""
        folder_id = self._get_or_create_encryption_bundles_folder_id()
        filename = self._encryption_bundle_filename(self.email, peer_email)
        file_payload, _ = self.client.create_file_payload(bundle_json)

        # Check if file already exists
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])

        if items:
            execute_with_retries(
                self.client.drive_service.files().update(
                    fileId=items[0]["id"], media_body=file_payload
                )
            )
        else:
            file_metadata = {"name": filename, "parents": [folder_id]}
            self.client.create_syft_file(file_metadata, file_payload)

    def share_encryption_bundles_folder(self, peer_email: str) -> None:
        """Share the bundles folder with a peer so they can read bundles."""
        folder_id = self._get_or_create_encryption_bundles_folder_id()
        self.client.add_permission(folder_id, peer_email, write=False)

    def read_own_encryption_bundle(self, peer_email: str) -> str | None:
        """Read the bundle this client published for ``peer_email``, or None.

        A read never creates the bundles folder: without it, nothing is published.
        """
        folder_id = self._find_encryption_bundles_folder_id()
        if folder_id is None:
            return None
        filename = self._encryption_bundle_filename(self.email, peer_email)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        if not items:
            return None
        return self.client.download_file(items[0]["id"]).decode("utf-8")

    def read_peer_encryption_bundle(self, peer_email: str) -> str | None:
        """Read encryption bundle that peer wrote for us.

        Searches for: encryption_bundle_{peer_email}_for_{self.email}.json
        in peer's bundles folder.
        """
        filename = self._encryption_bundle_filename(peer_email, self.email)
        query = f"name='{filename}' and trashed=false and '{peer_email}' in owners"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        if not items:
            return None
        try:
            data = self.client.download_file(items[0]["id"])
        except Exception as e:
            print(f"Warning: could not download the bundle of {peer_email}: {e}")
            return None
        try:
            return data.decode("utf-8")
        except ValueError as e:
            print(f"Warning: could not read the bundle of {peer_email}: {e}")
            return None
