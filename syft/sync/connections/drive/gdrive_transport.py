"""A Google Drive account as a syft connection."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import build_http

from syft.sync.connections.base_connection import (
    SyftboxPlatformConnection,
)
from syft.sync.connections.drive.drive_collections import DriveCollections
from syft.sync.connections.drive.drive_connections import drive_transport
from syft.sync.connections.drive.drive_owner_store import DriveOwnerStore
from syft.sync.connections.drive.drive_peering import DrivePeering
from syft.sync.connections.drive.gdrive_client import (
    SYFT_FILE_PROPERTY_KEY,
    SYFT_FILE_PROPERTY_VALUE,
    GDriveClient,
)
from syft.sync.connections.drive.gdrive_layout import (
    SYFTBOX_FOLDER,
    DriveLayout,
    valid_events_from_file_metadatas,
)
from syft.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
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
from syft.sync.transport.protocols import Transport
from syft.sync.utils.syftbox_utils import check_env

if TYPE_CHECKING:
    from syft.sync.connections.drive.grdrive_config import (
        GdriveConnectionConfig,
    )

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


class GDriveConnection(SyftboxPlatformConnection):
    """A Google Drive account as a syft connection.

    It builds the Drive service and the ``DriveLayout``, and gives out the
    transport jobs (``as_transport``) and copies for other threads. It also
    reads and archives messages, and cleans up the account for
    ``delete_syftbox``.
    """

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

    @property
    def _peering(self) -> DrivePeering:
        peering = self.as_transport().peering
        assert isinstance(peering, DrivePeering)
        return peering

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
        self._peering.copy_caches_to(other._peering)

    @property
    def environment(self) -> Environment:
        return check_env()

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

    def reset_caches(self):
        self.layout.reset_caches()
        self._owner_store.reset_caches()
        self._collections.reset_caches()
        self._peering.reset_caches()

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
