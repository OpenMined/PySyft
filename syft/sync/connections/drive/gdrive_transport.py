"""Google Drive Files transport layer implementation"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import build_http
from pydantic import BaseModel
from syft_migration import MigrationError

from syft.sync.connections.base_connection import (
    SyftboxPlatformConnection,
)
from syft.sync.connections.drive.drive_collections import DriveCollections
from syft.sync.connections.drive.drive_connections import drive_transport
from syft.sync.connections.drive.drive_owner_store import DriveOwnerStore
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
    valid_events_from_file_metadatas,
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
)
from syft.sync.messages.proposed_filechange import (
    FileNameParseError,
    MessageFileName,
    ProposedFileChangesMessage,
)
from syft.sync.transport.naming import (
    PEERS_META_KEY,
    SYFT_PEERS_VERSION,
)
from syft.sync.transport.protocols import Transport
from syft.sync.utils.syftbox_utils import check_env

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
    # The transport jobs over this connection, built once so each job keeps
    # one set of caches.
    _transport: Transport | None = None

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

    def as_transport(self) -> Transport:
        """This connection as the transport jobs. See ``drive_connections.py``."""
        if self._transport is None:
            self._transport = drive_transport(self)
        return self._transport

    @property
    def _owner_store(self) -> DriveOwnerStore:
        owner_store = self.as_transport().owner_store
        assert isinstance(owner_store, DriveOwnerStore)
        return owner_store

    @property
    def _collections(self) -> DriveCollections:
        collections = self.as_transport().collections
        assert isinstance(collections, DriveCollections)
        return collections

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
        self._owner_store.copy_caches_to(other._owner_store)
        self._collections.copy_caches_to(other._collections)
        other._encryption_bundles_folder_id = self._encryption_bundles_folder_id
        other._peers_json_cache = (
            dict(self._peers_json_cache) if self._peers_json_cache is not None else None
        )

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

        file_metadatas = self.layout.get_file_metadatas_from_folder(
            folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = valid_events_from_file_metadatas(file_metadatas)
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

        file_metadatas = self.layout.get_file_metadatas_from_folder(
            folder_id, since_timestamp=since_timestamp
        )
        valid_fname_objs = valid_events_from_file_metadatas(file_metadatas)
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
        file_metadatas = self.layout.get_file_metadatas_from_folder(inbox_folder_id)
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
        return self.layout.get_file_metadatas_from_folder(
            folder_id, since_timestamp=None
        )

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
        self._owner_store.reset_caches()
        self._collections.reset_caches()
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
