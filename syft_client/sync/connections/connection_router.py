from pydantic import BaseModel
from typing import TYPE_CHECKING, List, Optional
from syft_client.sync.connections.base_connection import (
    ConnectionConfig,
    FileCollection,
    SyftboxPlatformConnection,
)
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.sync.events.file_change_event import (
    FileChangeEventsMessage,
)
from syft_client.sync.checkpoints.checkpoint import Checkpoint, IncrementalCheckpoint
from syft_client.sync.checkpoints.rolling_state import RollingState
from syft_client.sync.peers.peer_store import PeerStore
from syft_client.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft_client.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft_client.sync.peers.peer import Peer, PeerState
from syft_client.sync.utils.print_utils import (
    print_peer_adding_to_platform,
    print_peer_added_to_platform,
)

if TYPE_CHECKING:
    from syft_client.sync.version.version_info import VersionInfo


class ConnectionRouter(BaseModel):
    connections: List[SyftboxPlatformConnection]

    peer_store: PeerStore

    @classmethod
    def from_configs(cls, email: str, connection_configs: List[ConnectionConfig]):
        return cls(
            connections=[
                SyftboxPlatformConnection.from_config(config)
                for config in connection_configs
            ],
            peer_store=PeerStore(email=email),
        )

    def add_connection(self, connection: SyftboxPlatformConnection):
        self.connections.append(connection)

    def get_authenticated_email(self) -> str:
        return self.connections[0].get_authenticated_email()

    def connection_for_send_message(self) -> SyftboxPlatformConnection:
        return self.connections[0]

    def connection_for_receive_message(self) -> SyftboxPlatformConnection:
        return self.connections[0]

    def copy_connection(
        self, connection: SyftboxPlatformConnection
    ) -> SyftboxPlatformConnection:
        if isinstance(connection, GDriveConnection):
            return connection.copy()
        else:
            return connection

    def connection_for_eventlog(
        self, create_new: bool = False
    ) -> SyftboxPlatformConnection:
        existing_connection = self.connections[0]
        if create_new:
            return self.copy_connection(existing_connection)
        else:
            return existing_connection

    def connection_for_datasite_watcher(self) -> SyftboxPlatformConnection:
        return self.connections[0]

    def connection_for_outbox(self) -> SyftboxPlatformConnection:
        return self.connections[0]

    def connection_for_own_syftbox(self) -> SyftboxPlatformConnection:
        return self.connections[0]

    # =========================================================================
    # MESSAGE SEND/RECEIVE (with encryption)
    # =========================================================================

    def watcher_send_proposed_file_changes_message(
        self, recipient: str, proposed_file_changes_message: ProposedFileChangesMessage
    ):
        data = proposed_file_changes_message.as_compressed_data()
        data = self.peer_store.encrypt_if_needed(recipient, data)
        filename = proposed_file_changes_message.message_filename.as_string()
        self.connection_for_send_message().watcher_send_raw_bytes_to_inbox(
            recipient, filename, data
        )

    def owner_get_next_proposed_filechange_message(
        self, sender_email: str
    ) -> ProposedFileChangesMessage | None:
        connection = self.connection_for_receive_message()
        result = connection.owner_download_next_raw_proposed_message_from_inbox(
            sender_email
        )
        if result is None:
            return None
        raw_data, file_id = result
        raw_data = self.peer_store.decrypt_and_verify_if_needed(sender_email, raw_data)
        msg = ProposedFileChangesMessage.from_compressed_data(raw_data)
        msg.platform_id = file_id
        return msg

    def owner_write_event_messages_to_outbox(
        self, recipient_email: str, events_message: FileChangeEventsMessage
    ):
        data = events_message.as_compressed_data()
        data = self.peer_store.encrypt_if_needed(recipient_email, data)
        fname = events_message.message_filepath.as_string()
        self.connection_for_outbox().owner_write_raw_bytes_to_outbox(
            recipient_email, fname, data
        )

    def watcher_get_events_messages(
        self, peer_email: str, since_timestamp: float | None
    ) -> List[FileChangeEventsMessage]:
        connection = self.connection_for_datasite_watcher()
        raw_list = connection.watcher_download_raw_events_from_outbox(
            peer_email, since_timestamp
        )
        return [
            FileChangeEventsMessage.from_compressed_data(
                self.peer_store.decrypt_and_verify_if_needed(peer_email, raw)
            )
            for raw in raw_list
        ]

    def watcher_get_outbox_file_metadatas(
        self, peer_email: str, since_timestamp: float | None
    ) -> List[dict]:
        connection = self.connection_for_datasite_watcher()
        return connection.watcher_get_outbox_file_metadatas(peer_email, since_timestamp)

    def connection_for_parallel_download(self) -> SyftboxPlatformConnection:
        """Create a new connection for thread-safe parallel downloads."""
        return self.copy_connection(self.connection_for_datasite_watcher())

    # =========================================================================
    # EVENT LOG (self-encrypted at rest when encryption enabled)
    # =========================================================================

    def owner_write_events_message_to_syftbox(
        self, events_message: FileChangeEventsMessage
    ):
        data = events_message.as_compressed_data()
        data = self.peer_store.encrypt_for_self_if_needed(data)
        filename = events_message.message_filepath.as_string()
        self.connection_for_eventlog().owner_write_raw_bytes_to_syftbox(filename, data)

    def owner_get_all_accepted_events_messages(self) -> List[FileChangeEventsMessage]:
        connection = self.connection_for_eventlog()
        raw_list = connection.owner_download_all_raw_events_from_syftbox()
        return [
            FileChangeEventsMessage.from_compressed_data(
                self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
            )
            for raw in raw_list
        ]

    def download_events_message_by_id(
        self, events_message_id: str
    ) -> FileChangeEventsMessage:
        """Download and decrypt a single events message (thread-safe)."""
        connection = self.connection_for_eventlog(create_new=True)
        raw = connection.owner_download_raw_bytes_by_id(events_message_id)
        raw = self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
        return FileChangeEventsMessage.from_compressed_data(raw)

    # =========================================================================
    # PEER MANAGEMENT
    # =========================================================================

    def add_peer(self, peer_email: str, verbose: bool = True) -> Peer:
        connection = self.connection_for_receive_message()
        platform = GdriveFilesPlatform()

        if verbose:
            print_peer_adding_to_platform(peer_email, platform.module_path)

        connection.add_peer(peer_email=peer_email)

        if verbose:
            print_peer_added_to_platform(peer_email, platform.module_path)
        return Peer(email=peer_email, platforms=[platform])

    def get_all_peers_from_json(self, force_download: bool = False) -> List[Peer]:
        """Get all peers from SYFT_peers.json with their stored state."""
        connection = self.connection_for_send_message()
        peers_data = connection._get_peers_json(force_download=force_download)
        peers = []
        for email, data in peers_data.items():
            try:
                state = PeerState(data.get("state", "unknown"))
            except ValueError:
                continue
            peer = Peer(
                email=email,
                platforms=[GdriveFilesPlatform()],
                state=state,
                public_encryption_bundle=data.get("public_encryption_bundle"),
            )
            peers.append(peer)
        return peers

    def get_peer_requests(self) -> List[Peer]:
        """Get pending peer requests"""
        connection = self.connection_for_send_message()
        peer_emails = connection.get_peer_requests()
        return [
            Peer(
                email=peer_email,
                platforms=[GdriveFilesPlatform()],
                state=PeerState.REQUESTED_BY_PEER,
            )
            for peer_email in peer_emails
        ]

    def update_peer_state(
        self,
        peer_email: str,
        state: str,
        public_encryption_bundle: dict | None = None,
    ):
        """Update peer state in storage"""
        connection = self.connection_for_send_message()
        connection._update_peer_state(peer_email, state, public_encryption_bundle)

    def owner_remove_proposed_filechange_from_inbox(
        self, proposed_filechange_message: ProposedFileChangesMessage
    ):
        connection = self.connection_for_receive_message()
        connection.owner_remove_proposed_filechange_message_from_inbox(
            proposed_filechange_message
        )

    # =========================================================================
    # ENCRYPTION BUNDLE EXCHANGE
    # =========================================================================

    def write_encryption_bundle(self, peer_email: str, bundle_json: str) -> None:
        """Write own encryption bundle for a peer."""
        connection = self.connection_for_own_syftbox()
        connection.write_encryption_bundle(peer_email, bundle_json)

    def share_encryption_bundles_folder(self, peer_email: str) -> None:
        """Share the bundles folder with a peer."""
        connection = self.connection_for_own_syftbox()
        connection.share_encryption_bundles_folder(peer_email)

    def read_peer_encryption_bundle(self, peer_email: str) -> str | None:
        """Read encryption bundle that peer wrote for us."""
        connection = self.connection_for_datasite_watcher()
        return connection.read_peer_encryption_bundle(peer_email)

    # =========================================================================
    # DATASET METHODS (with encryption)
    # =========================================================================

    def owner_create_dataset_collection_folder(
        self, tag: str, content_hash: str, owner_email: str
    ) -> str:
        connection = self.connection_for_send_message()
        return connection.owner_create_dataset_collection_folder(
            tag, content_hash, owner_email
        )

    def owner_tag_dataset_collection_as_any(self, tag: str, content_hash: str) -> None:
        connection = self.connection_for_send_message()
        connection.owner_tag_dataset_collection_as_any(tag, content_hash)

    def owner_share_dataset_collection(
        self, tag: str, content_hash: str, users: list[str]
    ) -> None:
        connection = self.connection_for_send_message()
        connection.owner_share_dataset_collection(tag, content_hash, users)

    def owner_upload_dataset_files(
        self,
        tag: str,
        content_hash: str,
        files: dict[str, bytes],
        recipient_email: str | None = None,
    ) -> None:
        """Upload dataset files, encrypting each file if encryption is enabled."""
        if recipient_email and self.peer_store:
            files = {
                name: self.peer_store.encrypt_if_needed(recipient_email, data)
                for name, data in files.items()
            }
        connection = self.connection_for_send_message()
        connection.owner_upload_dataset_files(tag, content_hash, files)

    def owner_list_dataset_collections(self) -> list[str]:
        connection = self.connection_for_send_message()
        return connection.owner_list_dataset_collections()

    def owner_list_all_dataset_collections_with_permissions(
        self,
    ) -> list[FileCollection]:
        connection = self.connection_for_send_message()
        return connection.owner_list_all_dataset_collections_with_permissions()

    def owner_delete_dataset_collection(self, tag: str) -> None:
        connection = self.connection_for_send_message()
        connection.owner_delete_dataset_collection(tag)

    def owner_delete_private_dataset_collection(self, tag: str) -> None:
        connection = self.connection_for_send_message()
        connection.owner_delete_private_dataset_collection(tag)

    def watcher_list_dataset_collections(self) -> list[dict]:
        connection = self.connection_for_receive_message()
        return connection.watcher_list_dataset_collections()

    def watcher_download_dataset_collection(
        self, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        connection = self.connection_for_datasite_watcher()
        files = connection.watcher_download_dataset_collection(
            tag, content_hash, owner_email
        )
        if self.peer_store and owner_email:
            files = {
                name: self.peer_store.decrypt_dataset_if_needed(owner_email, data)
                for name, data in files.items()
            }
        return files

    def owner_create_private_dataset_collection_folder(
        self, tag: str, content_hash: str, owner_email: str
    ) -> str:
        connection = self.connection_for_send_message()
        return connection.owner_create_private_dataset_collection_folder(
            tag, content_hash, owner_email
        )

    def owner_upload_private_dataset_files(
        self, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        connection = self.connection_for_send_message()
        connection.owner_upload_private_dataset_files(tag, content_hash, files)

    def owner_list_private_dataset_collections(self) -> list[FileCollection]:
        connection = self.connection_for_send_message()
        return connection.owner_list_private_dataset_collections()

    def owner_get_private_collection_file_metadatas(
        self, tag: str, content_hash: str, owner_email: str
    ) -> List[dict]:
        connection = self.connection_for_datasite_watcher()
        return connection.owner_get_private_collection_file_metadatas(
            tag, content_hash, owner_email
        )

    def connection_for_version_read(
        self, create_new: bool = False
    ) -> SyftboxPlatformConnection:
        """Get connection for reading version files. Can create new for thread safety."""
        existing_connection = self.connections[0]
        if create_new:
            return self.copy_connection(existing_connection)
        else:
            return existing_connection

    def write_version_file(self, version_info: "VersionInfo") -> None:
        """Write version file to own SyftBox folder."""
        connection = self.connection_for_own_syftbox()
        connection.write_version_file(version_info)

    def read_own_version_file(self) -> Optional["VersionInfo"]:
        """Read version file from own SyftBox folder."""
        connection = self.connection_for_own_syftbox()
        return connection.read_own_version_file()

    def read_peer_version_file(self, peer_email: str) -> Optional["VersionInfo"]:
        """Read version file from a peer's SyftBox folder."""
        connection = self.connection_for_datasite_watcher()
        return connection.read_peer_version_file(peer_email)

    def share_version_file_with_peer(self, peer_email: str) -> None:
        """Share version file with a peer so they can read it."""
        connection = self.connection_for_own_syftbox()
        connection.share_version_file_with_peer(peer_email)

    def watcher_get_dataset_collection_file_metadatas(
        self, tag: str, content_hash: str, owner_email: str
    ) -> List[dict]:
        connection = self.connection_for_datasite_watcher()
        return connection.watcher_get_dataset_collection_file_metadatas(
            tag, content_hash, owner_email
        )

    def watcher_download_dataset_file(self, file_id: str, owner_email: str) -> bytes:
        connection = self.connection_for_datasite_watcher()
        data = connection.watcher_download_dataset_file(file_id)
        data = self.peer_store.decrypt_dataset_if_needed(owner_email, data)
        return data

    # =========================================================================
    # MISC
    # =========================================================================

    def delete_multiple_files_by_ids(
        self,
        file_ids: List[str],
        ignore_permissions_errors: bool = True,
        ignore_file_not_found: bool = True,
    ):
        connection = self.connection_for_own_syftbox()
        connection.delete_multiple_files_by_ids(
            file_ids,
            ignore_permissions_errors=ignore_permissions_errors,
            ignore_file_not_found=ignore_file_not_found,
        )

    def owner_get_all_accepted_event_file_ids(
        self, since_timestamp: float | None = None
    ) -> List[str]:
        connection = self.connection_for_eventlog()
        return connection.owner_get_all_accepted_event_file_ids(since_timestamp)

    def gather_all_file_and_folder_ids(self) -> List[str]:
        connection = self.connection_for_own_syftbox()
        return connection.gather_all_file_and_folder_ids()

    def find_orphaned_message_files(self) -> List[str]:
        """Find message files by name pattern (catches orphaned files)."""
        connection = self.connection_for_own_syftbox()
        return connection.find_orphaned_message_files()

    def reset_caches(self):
        connection = self.connection_for_own_syftbox()
        connection.reset_caches()

    def delete_file_by_id(self, file_id: str):
        connection = self.connection_for_own_syftbox()
        connection.delete_file_by_id(file_id)

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def upload_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize, encrypt, and upload a checkpoint."""
        data = checkpoint.as_compressed_data()
        data = self.peer_store.encrypt_for_self_if_needed(data)
        connection = self.connection_for_own_syftbox()
        return connection.upload_raw_checkpoint(checkpoint.filename, data)

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Download, decrypt, and deserialize the latest checkpoint."""
        connection = self.connection_for_own_syftbox()
        raw = connection.download_raw_latest_checkpoint()
        if raw is None:
            return None
        try:
            raw = self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
            return Checkpoint.from_compressed_data(raw)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def get_events_count_since_checkpoint(
        self, checkpoint_timestamp: float | None
    ) -> int:
        """Count events created after the checkpoint timestamp."""
        connection = self.connection_for_eventlog()
        return connection.get_events_count_since_checkpoint(checkpoint_timestamp)

    def get_events_messages_since_timestamp(
        self, since_timestamp: float
    ) -> List[FileChangeEventsMessage]:
        """Download, decrypt, and deserialize events since a timestamp."""
        connection = self.connection_for_eventlog()
        raw_list = connection.download_raw_events_since_timestamp(since_timestamp)
        return [
            FileChangeEventsMessage.from_compressed_data(
                self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
            )
            for raw in raw_list
        ]

    # =========================================================================
    # INCREMENTAL CHECKPOINT METHODS
    # =========================================================================

    def upload_incremental_checkpoint(self, checkpoint: IncrementalCheckpoint) -> str:
        """Serialize, encrypt, and upload an incremental checkpoint."""
        data = checkpoint.as_compressed_data()
        data = self.peer_store.encrypt_for_self_if_needed(data)
        connection = self.connection_for_own_syftbox()
        return connection.upload_raw_incremental_checkpoint(checkpoint.filename, data)

    def get_all_incremental_checkpoints(self) -> List[IncrementalCheckpoint]:
        """Download, decrypt, and deserialize all incremental checkpoints."""
        connection = self.connection_for_own_syftbox()
        raw_list = connection.download_all_raw_incremental_checkpoints()
        checkpoints = []
        for raw in raw_list:
            try:
                raw = self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
                cp = IncrementalCheckpoint.from_compressed_data(raw)
                checkpoints.append(cp)
            except Exception as e:
                print(f"Warning: Failed to load incremental checkpoint: {e}")
                continue
        return sorted(checkpoints, key=lambda c: c.sequence_number)

    def get_incremental_checkpoint_count(self) -> int:
        """Get the number of incremental checkpoints."""
        connection = self.connection_for_own_syftbox()
        return connection.get_incremental_checkpoint_count()

    def get_next_incremental_sequence_number(self) -> int:
        """Get the next sequence number for incremental checkpoints."""
        connection = self.connection_for_own_syftbox()
        return connection.get_next_incremental_sequence_number()

    def delete_all_incremental_checkpoints(self) -> None:
        """Delete all incremental checkpoints from the storage backend."""
        connection = self.connection_for_own_syftbox()
        connection.delete_all_incremental_checkpoints()

    # =========================================================================
    # ROLLING STATE METHODS
    # =========================================================================

    def upload_rolling_state(self, rolling_state: RollingState) -> str:
        """Serialize, encrypt, and upload rolling state."""
        data = rolling_state.as_compressed_data()
        data = self.peer_store.encrypt_for_self_if_needed(data)
        connection = self.connection_for_own_syftbox()
        return connection.upload_raw_rolling_state(rolling_state.filename, data)

    def get_rolling_state(self) -> RollingState | None:
        """Download, decrypt, and deserialize rolling state."""
        connection = self.connection_for_own_syftbox()
        raw = connection.download_raw_rolling_state()
        if raw is None:
            return None
        try:
            raw = self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
            return RollingState.from_compressed_data(raw)
        except Exception as e:
            print(f"Warning: Failed to load rolling state: {e}")
            return None

    def delete_rolling_state(self) -> None:
        """Delete rolling state from the storage backend."""
        connection = self.connection_for_own_syftbox()
        connection.delete_rolling_state()
