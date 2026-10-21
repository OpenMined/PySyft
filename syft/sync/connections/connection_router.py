import logging
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from pydantic import BaseModel, PrivateAttr
from syft_migration import MigratableObject, ProtocolSchema

from syft.migrations import (
    SYFT_CLIENT_PROTOCOL_VERSION,
    client_migration_service,
    client_registry,
)
from syft.sync.checkpoints.checkpoint import Checkpoint, IncrementalCheckpoint
from syft.sync.checkpoints.rolling_state import RollingState
from syft.sync.connections.base_connection import (
    ConnectionConfig,
    FileCollection,
    SyftboxPlatformConnection,
)
from syft.sync.encryption_policy import (
    EncryptionPolicyConflictError,
    PeerEncryptionPendingError,
    UnencryptedMessageError,
)
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)
from syft.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft.sync.peers.peer import Peer, PeerState
from syft.sync.peers.peer_store import PeerStore
from syft.sync.transport.naming import PEERS_META_KEY
from syft.sync.transport.protocols import Transport
from syft.sync.transport.types import MessageKind, SendStatus
from syft.sync.utils.print_utils import (
    print_peer_added_to_platform,
    print_peer_adding_to_platform,
)

if TYPE_CHECKING:
    from syft.sync.version.version_info import VersionInfo


logger = logging.getLogger(__name__)


class ConnectionRouter(BaseModel):
    connections: List[SyftboxPlatformConnection]

    peer_store: PeerStore

    # peer email -> syft ProtocolSchema; syft wires PeerManager's
    # live map here (updated in place as peer version files load), so outgoing
    # messages downgrade to what each peer reads.
    _peer_schemas: Dict[str, ProtocolSchema] = PrivateAttr(default_factory=dict)
    # Reads the version file of a peer and updates what depends on it (the
    # peer, its policy, the protocol map). Set by SyftboxManager to
    # ``PeerManager.load_peer_version``.
    _version_loader: Optional[Callable[[str], object]] = PrivateAttr(default=None)

    def set_version_loader(self, version_loader: Callable[[str], object]) -> None:
        self._version_loader = version_loader

    # The transport jobs over connections[0], rebuilt if that connection is
    # replaced (tests swap in a mock connection with add_connection).
    _transport: Optional[Transport] = PrivateAttr(default=None)
    _transport_source: Optional[SyftboxPlatformConnection] = PrivateAttr(default=None)

    @property
    def transport(self) -> Transport:
        connection = self.connections[0]
        if self._transport is None or self._transport_source is not connection:
            self._transport = connection.as_transport()
            self._transport_source = connection
        return self._transport

    def set_peer_schemas(self, peer_schemas: Dict[str, ProtocolSchema]) -> None:
        """Adopt the live {peer email -> syft ProtocolSchema} map."""
        self._peer_schemas = peer_schemas

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
        return connection.clone_for_thread()

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

    def _downgrade_for_peer(
        self, message: MigratableObject, peer_email: str
    ) -> MigratableObject:
        """Downgrade an outgoing message to the protocol the peer reads.

        The receive paths upgrade every blob on read, so this is the other
        half of the contract: both sides speak the lower of the two protocol
        versions, bounded by both floors. Migrations return new objects, so
        the caller's message is never mutated (one instance fans out to many
        recipients).

        A peer without a known schema is assumed to run the current protocol,
        the same policy as jobs; the assumption is logged.
        """
        schema = self._peer_schemas.get(peer_email)
        if schema is None:
            logger.warning(
                f"No syft protocol schema known for peer {peer_email!r}. "
                f"This client writes protocol {SYFT_CLIENT_PROTOCOL_VERSION}. A "
                "peer that speaks an earlier protocol cannot read this message."
            )
            return message
        protocol_version = client_registry.infer_protocol_version_to_use(
            peer_version=schema.version,
            peer_min=schema.min_supported_version,
        )
        if schema.version == protocol_version:
            # The peer speaks the negotiated version, so its advertised slim
            # schema is the target; computing our own full schema on every
            # send would rebuild every object's JSON schema for nothing.
            target = schema
        else:
            target = client_registry.schema_for_protocol_version(protocol_version)
        return client_migration_service.migrate_to_schema(message, target)

    def _refresh_unknown_peer(self, peer_email: str) -> None:
        """Read the version file of ``peer_email`` again when its policy is not known.

        The peer can have accepted since the last load. It runs at most once
        per peer load, and before a message is converted for the peer.
        """
        if self.peer_store.peer_policy_known(peer_email):
            return
        peer = self.peer_store.get_cached_peer(peer_email)
        if peer is None or not self.peer_store.take_policy_reread(peer_email):
            return
        if self._version_loader is not None:
            self._version_loader(peer_email)
        else:
            peer.version = self.read_peer_version_file(peer_email)

    def watcher_send_proposed_file_changes_message(
        self, recipient: str, proposed_file_changes_message: ProposedFileChangesMessage
    ):
        self._refresh_unknown_peer(recipient)
        proposed_file_changes_message = self._downgrade_for_peer(
            proposed_file_changes_message, recipient
        )
        data = proposed_file_changes_message.as_compressed_data()
        data = self.peer_store.encrypt_if_needed(recipient, data)
        filename = proposed_file_changes_message.message_filename.as_string()
        self.transport.p2p.send(recipient, filename, data, MessageKind.PROPOSED_CHANGES)

    def owner_get_next_proposed_filechange_message(
        self, sender_email: str
    ) -> ProposedFileChangesMessage | None:
        """The next message from ``sender_email`` that can be read, or None.

        A plaintext message to a pair that encrypts is archived unread, with a
        warning. None also means the messages of this peer wait (see
        ``_decrypt_or_wait``).
        """
        connection = self.connection_for_receive_message()
        while True:
            result = connection.owner_download_next_raw_proposed_message_from_inbox(
                sender_email
            )
            if result is None:
                return None
            raw_data, file_id = result
            try:
                decrypted = self._decrypt_or_wait(sender_email, raw_data)
            except UnencryptedMessageError as e:
                warnings.warn(f"Archived message file {file_id} unread: {e}")
                msg = ProposedFileChangesMessage.from_compressed_data(raw_data)
                msg.platform_id = file_id
                connection.owner_remove_proposed_filechange_message_from_inbox(
                    msg, sender_email
                )
                continue
            if decrypted is None:
                return None
            msg = ProposedFileChangesMessage.from_compressed_data(decrypted)
            msg.platform_id = file_id
            return msg

    def owner_write_event_messages_to_outbox(
        self, recipient_email: str, events_message: FileChangeEventsMessage
    ) -> SendStatus:
        self._refresh_unknown_peer(recipient_email)
        events_message = self._downgrade_for_peer(events_message, recipient_email)
        data = events_message.as_compressed_data()
        data = self.peer_store.encrypt_if_needed(recipient_email, data)
        fname = events_message.message_filepath.as_string()
        return self.transport.p2p.send(
            recipient_email, fname, data, MessageKind.FILE_EVENTS
        )

    def watcher_get_events_messages(
        self, peer_email: str, since_timestamp: float | None
    ) -> List[FileChangeEventsMessage]:
        """The outbox messages of ``peer_email``, up to the first that cannot be read yet."""
        connection = self.connection_for_datasite_watcher()
        messages = []
        for meta in connection.watcher_get_outbox_file_metadatas(
            peer_email, since_timestamp
        ):
            message = self.watcher_read_events_message(connection, peer_email, meta)
            if message is None:
                break
            messages.append(message)
        return messages

    def watcher_read_events_message(
        self,
        connection: SyftboxPlatformConnection,
        peer_email: str,
        file_meta: dict,
    ) -> FileChangeEventsMessage | None:
        """Download, verify and decrypt one outbox events message.

        ``file_meta`` holds its ``file_id`` and ``file_name``. Returns None
        when the message cannot be read yet (see ``_decrypt_or_wait``). A
        plaintext message to a pair that encrypts is skipped with a warning:
        an empty message takes its place, so later messages apply.
        """
        raw = connection.download_file(file_meta["file_id"])
        try:
            raw = self._decrypt_or_wait(peer_email, raw)
        except UnencryptedMessageError as e:
            warnings.warn(f"Skipped outbox message {file_meta['file_name']}: {e}")
            return FileChangeEventsMessage(
                events=[],
                message_filepath=FileChangeEventsMessageFileName.from_string(
                    file_meta["file_name"]
                ),
            )
        if raw is None:
            return None
        return FileChangeEventsMessage.from_compressed_data(raw)

    def _decrypt_or_wait(self, peer_email: str, raw: bytes) -> bytes | None:
        """Verify and decrypt an inbox or outbox message from ``peer_email``.

        Returns None when the message cannot be read yet; the messages of this
        peer then wait, so they keep their order.

        Raises:
            UnencryptedMessageError: when the pair encrypts and ``raw`` is
                plaintext.
        """
        try:
            return self.peer_store.decrypt_and_verify_if_needed(peer_email, raw)
        except (PeerEncryptionPendingError, EncryptionPolicyConflictError) as e:
            # A conflict found here is new since the peer filter of this sync.
            warnings.warn(f"Messages from {peer_email} wait: {e}")
        return None

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
        self.transport.owner_store.append_event(filename, data)

    def owner_get_all_accepted_events_messages(self) -> List[FileChangeEventsMessage]:
        raw_list = self.transport.owner_store.fetch_all_events()
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
        raw = connection.as_transport().owner_store.fetch(events_message_id)
        raw = self.peer_store.decrypt_and_verify_for_self_if_needed(raw)
        return FileChangeEventsMessage.from_compressed_data(raw)

    # =========================================================================
    # PEER MANAGEMENT
    # =========================================================================

    def add_peer(self, peer_email: str, verbose: bool = True) -> Peer:
        platform = self.transport.platform

        if verbose:
            print_peer_adding_to_platform(peer_email, platform.module_path)

        self.transport.peering.open_channels(peer_email)

        if verbose:
            print_peer_added_to_platform(peer_email, platform.module_path)
        return Peer(email=peer_email, platforms=[platform])

    def get_all_peers_from_json(self, force_download: bool = False) -> List[Peer]:
        """Get all peers from SYFT_peers.json with their stored state."""
        peers_data = self.transport.peering.read_peer_registry(
            force_download=force_download
        )
        peers = []
        for email, data in peers_data.items():
            if email == PEERS_META_KEY:
                continue
            try:
                state = PeerState(data.get("state", "unknown"))
            except ValueError:
                # A later client wrote a state that this client does not know.
                # The writer changes one entry and keeps the rest, so the entry
                # stays in the file. The peer returns after an upgrade.
                logger.warning(
                    f"Skipping peer {email}: unknown state "
                    f"{data.get('state')!r}. Install a newer syft to see "
                    "this peer."
                )
                continue
            peer = Peer(
                email=email,
                platforms=[self.transport.platform],
                state=state,
                public_encryption_bundle=data.get("public_encryption_bundle"),
            )
            peers.append(peer)
        return peers

    def get_peer_requests(self) -> List[Peer]:
        """Get pending peer requests"""
        peer_emails = self.transport.peering.incoming_requests()
        return [
            Peer(
                email=peer_email,
                platforms=[self.transport.platform],
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
        self.transport.peering.update_peer_registry(
            peer_email, state, public_encryption_bundle
        )

    def owner_remove_proposed_filechange_from_inbox(
        self,
        proposed_filechange_message: ProposedFileChangesMessage,
        sender_email: str,
    ):
        connection = self.connection_for_receive_message()
        connection.owner_remove_proposed_filechange_message_from_inbox(
            proposed_filechange_message, sender_email
        )

    # =========================================================================
    # ENCRYPTION BUNDLE EXCHANGE
    # =========================================================================

    def read_own_encryption_bundle(self, peer_email: str) -> str | None:
        """Read the bundle this client published for ``peer_email``."""
        return self.transport.peering.read_own_key_bundle(peer_email)

    def write_encryption_bundle(self, peer_email: str, bundle_json: str) -> None:
        """Write own encryption bundle for a peer."""
        self.transport.peering.publish_key_bundle(peer_email, bundle_json)

    def share_encryption_bundles_folder(self, peer_email: str) -> None:
        """Share the bundles folder with a peer."""
        self.transport.peering.grant_key_bundle_read(peer_email)

    def read_peer_encryption_bundle(self, peer_email: str) -> str | None:
        """Read encryption bundle that peer wrote for us."""
        return self.transport.peering.read_peer_key_bundle(peer_email)

    # =========================================================================
    # GENERIC PREFIX-PARAMETERIZED COLLECTION METHODS (with encryption)
    # =========================================================================

    def owner_create_collection_folder(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> str:
        return self.transport.collections.create(prefix, tag, content_hash, owner_email)

    def owner_tag_collection_as_any(
        self, prefix: str, tag: str, content_hash: str
    ) -> None:
        self.transport.collections.mark_shared_with_any(prefix, tag, content_hash)

    def owner_share_collection(
        self, prefix: str, tag: str, content_hash: str, users: list[str]
    ) -> None:
        self.transport.collections.grant_read(prefix, tag, content_hash, users)

    def owner_upload_collection_files(
        self,
        prefix: str,
        tag: str,
        content_hash: str,
        files: dict[str, bytes],
        recipient_email: str | None = None,
    ) -> None:
        """Upload collection files, encrypting each file if encryption is enabled."""
        if recipient_email and self.peer_store:
            self._refresh_unknown_peer(recipient_email)
            files = {
                name: self.peer_store.encrypt_if_needed(recipient_email, data)
                for name, data in files.items()
            }
        self.transport.collections.upload(prefix, tag, content_hash, files)

    def owner_list_collections(self, prefix: str) -> list[str]:
        return self.transport.collections.list_mine(prefix)

    def owner_list_all_collections_with_permissions(
        self, prefix: str
    ) -> list[FileCollection]:
        return self.transport.collections.list_mine_with_permissions(prefix)

    def owner_delete_collection(self, prefix: str, tag: str) -> None:
        self.transport.collections.delete(prefix, tag)

    def watcher_list_collections(self, prefix: str) -> list[dict]:
        return self.transport.collections.list_shared_with_me(prefix)

    def watcher_download_collection(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        files = self.transport.collections.download(
            prefix, tag, content_hash, owner_email
        )
        if self.peer_store and owner_email:
            files = {
                name: self.peer_store.decrypt_dataset_if_needed(owner_email, data)
                for name, data in files.items()
            }
        return files

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
        self.transport.peering.publish_version(version_info)

    def read_own_version_file(self) -> Optional["VersionInfo"]:
        """Read version file from own SyftBox folder."""
        return self.transport.peering.read_own_version()

    def read_peer_version_file(self, peer_email: str) -> Optional["VersionInfo"]:
        """Read version file from a peer's SyftBox folder."""
        return self.transport.peering.read_peer_version(peer_email)

    def share_version_file_with_peer(
        self, peer_email: str, version_info: Optional["VersionInfo"] = None
    ) -> None:
        """Share version file with a peer so they can read it."""
        self.transport.peering.grant_version_read(peer_email, version_info)

    def watcher_get_collection_file_metadatas(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> List[dict]:
        return self.transport.collections.list_files(
            prefix, tag, content_hash, owner_email
        )

    def watcher_download_collection_file(self, file_id: str, owner_email: str) -> bytes:
        data = self.transport.collections.fetch_file(file_id)
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
        return self.transport.owner_store.list_event_handles(since_timestamp)

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
        return self.transport.owner_store.put_checkpoint(checkpoint.filename, data)

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Download, decrypt, and deserialize the latest checkpoint."""
        raw = self.transport.owner_store.latest_checkpoint()
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
        return self.transport.owner_store.count_events_since(checkpoint_timestamp)

    def get_events_messages_since_timestamp(
        self, since_timestamp: float
    ) -> List[FileChangeEventsMessage]:
        """Download, decrypt, and deserialize events since a timestamp."""
        raw_list = self.transport.owner_store.fetch_events_since(since_timestamp)
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
        return self.transport.owner_store.put_incremental(checkpoint.filename, data)

    def get_all_incremental_checkpoints(self) -> List[IncrementalCheckpoint]:
        """Download, decrypt, and deserialize all incremental checkpoints."""
        raw_list = self.transport.owner_store.all_incrementals()
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
        return self.transport.owner_store.incremental_count()

    def get_next_incremental_sequence_number(self) -> int:
        """Get the next sequence number for incremental checkpoints."""
        return self.transport.owner_store.next_incremental_sequence()

    def delete_all_incremental_checkpoints(self) -> None:
        """Delete all incremental checkpoints from the storage backend."""
        self.transport.owner_store.delete_incrementals()

    # =========================================================================
    # ROLLING STATE METHODS
    # =========================================================================

    def upload_rolling_state(self, rolling_state: RollingState) -> str:
        """Serialize, encrypt, and upload rolling state."""
        data = rolling_state.as_compressed_data()
        data = self.peer_store.encrypt_for_self_if_needed(data)
        return self.transport.owner_store.put_rolling_state(
            rolling_state.filename, data
        )

    def get_rolling_state(self) -> RollingState | None:
        """Download, decrypt, and deserialize rolling state."""
        raw = self.transport.owner_store.get_rolling_state()
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
        self.transport.owner_store.delete_rolling_state()
