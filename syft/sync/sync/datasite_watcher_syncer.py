import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List

from pydantic import BaseModel, Field, PrivateAttr

from syft.sync.connections.base_connection import ConnectionConfig
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.callback_mixin import BaseModelCallbackMixin
from syft.sync.encryption_policy import (
    EncryptionPolicyConflictError,
    PeerEncryptionPendingError,
    PeerEncryptionPolicyUnknownError,
)
from syft.sync.events.file_change_event import FileChangeEventsMessage
from syft.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
)
from syft.sync.sync.constants import CACHE_DIR, HELD_PROPOSALS_FILENAME
from syft.sync.sync.held_messages import HeldMessages
from syft.sync.sync.caches.datasite_watcher_cache import (
    DataSiteWatcherCache,
    DataSiteWatcherCacheConfig,
)

logger = logging.getLogger(__name__)


class DatasiteWatcherSyncerConfig(BaseModel):
    syftbox_folder: Path | None = None
    email: str | None = None
    connection_configs: List[ConnectionConfig] = []
    datasite_watcher_cache_config: DataSiteWatcherCacheConfig = Field(
        default_factory=DataSiteWatcherCacheConfig
    )


class DatasiteWatcherSyncer(BaseModelCallbackMixin):
    """Handles both pushing proposed file changes and pulling from datasite outboxes."""

    class Config:
        arbitrary_types_allowed = True

    syftbox_folder: Path
    email: str
    connection_router: ConnectionRouter
    datasite_watcher_cache: DataSiteWatcherCache
    queue: Queue = Field(default_factory=Queue)

    _executor: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=10)
    )

    @classmethod
    def from_config(cls, config: DatasiteWatcherSyncerConfig):
        return cls(
            syftbox_folder=config.syftbox_folder,
            email=config.email,
            connection_router=ConnectionRouter.from_configs(
                config.email, config.connection_configs
            ),
            datasite_watcher_cache=DataSiteWatcherCache.from_config(
                config.datasite_watcher_cache_config
            ),
        )

    # --- Pushing proposed file changes ---

    def get_proposed_file_change_object(
        self,
        relative_path: Path,
        content: str,
        datasite_email: str | None = None,
        held_hashes: Dict[Path, int | None] | None = None,
    ) -> ProposedFileChange:
        if datasite_email is None:
            datasite_email = self.email
        cache_key = Path(datasite_email) / relative_path
        # The cache holds only sent changes. A held change to the same file
        # gives the next change its old hash.
        if held_hashes is None:
            held_hashes = self._held_hashes()
        if cache_key in held_hashes:
            old_hash = held_hashes[cache_key]
        else:
            old_hash = self.datasite_watcher_cache.current_hash_for_file(cache_key)
        return ProposedFileChange(
            datasite_email=datasite_email,
            path_in_datasite=relative_path,
            content=content,
            old_hash=old_hash,
        )

    def process_file_changes_queue(self):
        # One message per datasite: each change goes to the owner of its path.
        changes_by_recipient: Dict[str, List[ProposedFileChange]] = {}
        held_hashes: Dict[Path, int | None] | None = None
        while not self.queue.empty():
            relative_path, content = self.queue.get()

            # for in memory connection we pass content directly
            if content is None:
                with open(self.syftbox_folder / relative_path, "r") as f:
                    content = f.read()

            datasite_email = relative_path.parts[0]
            path_in_datasite = (
                Path(*relative_path.parts[1:])
                if len(relative_path.parts) > 1
                else Path()
            )

            # TODO: add some better parsing logic here
            recipient = datasite_email
            path_in_datasite = path_in_datasite

            if held_hashes is None:
                held_hashes = self._held_hashes()
            file_change = self.get_proposed_file_change_object(
                path_in_datasite,
                content,
                datasite_email=recipient,
                held_hashes=held_hashes,
            )
            changes_by_recipient.setdefault(recipient, []).append(file_change)

        error = None
        for recipient, file_changes in changes_by_recipient.items():
            message = ProposedFileChangesMessage(
                sender_email=self.email, proposed_file_changes=file_changes
            )
            try:
                sent = self._send_or_hold(recipient, message)
            except (
                EncryptionPolicyConflictError,
                PeerEncryptionPolicyUnknownError,
            ) as e:
                # The other recipients still get their changes.
                error = error or e
                continue
            if sent:
                self._record_sent_hashes(message)
        if error is not None:
            raise error

    def _record_sent_hashes(self, message: ProposedFileChangesMessage) -> None:
        hashes = self.datasite_watcher_cache.file_hashes
        for fc in message.proposed_file_changes:
            path_key = Path(fc.datasite_email) / fc.path_in_datasite
            if fc.is_deleted:
                hashes.pop(path_key, None)
            else:
                hashes[path_key] = fc.new_hash

    @property
    def _held_messages(self) -> HeldMessages:
        return HeldMessages(self.syftbox_folder / CACHE_DIR / HELD_PROPOSALS_FILENAME)

    def _held_hashes(self) -> Dict[Path, int | None]:
        """The new hash of each file that a held change touches."""
        messages = [
            ProposedFileChangesMessage.from_compressed_data(d)
            for _, d in self._held_messages.read()
        ]
        hashes: Dict[Path, int | None] = {}
        for message in messages:
            for fc in message.proposed_file_changes:
                path_key = Path(fc.datasite_email) / fc.path_in_datasite
                hashes[path_key] = None if fc.is_deleted else fc.new_hash
        return hashes

    def _send_or_hold(
        self, recipient: str, message: ProposedFileChangesMessage
    ) -> bool:
        """Send ``message``, or hold it while encryption with the recipient is pending.

        Encryption is pending while the recipient policy is unknown, or while
        the pair encrypts and the keys or the peer bundle are missing. A held
        message is kept on disk. Messages to one recipient keep their order.
        Returns True when sent.

        Raises:
            PeerEncryptionPolicyUnknownError: when ``recipient`` is not a peer.
                It never gets a policy, so a held message would never go.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        with self._held_messages.edit() as held:
            if not any(r == recipient for r, _ in held):
                try:
                    self.connection_router.watcher_send_proposed_file_changes_message(
                        recipient, message
                    )
                    return True
                except PeerEncryptionPendingError as e:
                    peer_store = self.connection_router.peer_store
                    if peer_store.peers_loaded and not peer_store.get_cached_peer(
                        recipient
                    ):
                        raise
                    print(f"Held a message for {recipient}: {e}")
            held.append((recipient, message.as_compressed_data()))
            return False

    @property
    def pending_message_count(self) -> int:
        return len(self._held_messages)

    def send_pending_messages(self) -> None:
        """Send the held messages whose recipient policy is now known.

        Messages for a recipient that is no longer a peer (the peer request
        was rejected), or whose policy conflicts, are dropped with a warning.
        """
        peer_store = self.connection_router.peer_store
        store = self._held_messages
        with store.edit() as held:
            waiting: set[str] = set()
            for item in list(held):
                recipient, data = item
                if recipient in waiting:
                    # Keep the order of the messages to one recipient.
                    continue
                if not peer_store.peers_loaded:
                    waiting.add(recipient)
                    continue
                if peer_store.get_cached_peer(recipient) is None:
                    logger.warning(
                        f"Dropped a held message for {recipient}: it is not a peer."
                    )
                else:
                    message = ProposedFileChangesMessage.from_compressed_data(data)
                    try:
                        self.connection_router.watcher_send_proposed_file_changes_message(
                            recipient, message
                        )
                    except PeerEncryptionPendingError:
                        waiting.add(recipient)
                        continue
                    except EncryptionPolicyConflictError as e:
                        print(f"Dropped a held message for {recipient}: {e}")
                    else:
                        self._record_sent_hashes(message)
                # Saved after each message, so an error part way leaves
                # exactly the messages that were not sent.
                held.remove(item)
                store.save(held)

    def resend_rejected_messages(self, peer_emails: list[str]) -> None:
        """Resend, encrypted, the plaintext messages that peers rejected.

        A peer rejects plaintext when it requires encryption and this client
        sent with an outdated view of its policy.
        """
        for peer_email in peer_emails:
            count = self.connection_router.watcher_resend_rejected_inbox_messages(
                peer_email
            )
            if count:
                logger.info(f"Resent {count} rejected message(s) to {peer_email}")

    def on_file_change(
        self, relative_path: Path | str, content: str | None = None, process_now=True
    ):
        relative_path = Path(relative_path)
        self.queue.put((relative_path, content))
        if process_now:
            self.process_file_changes_queue()

    # --- Pulling from datasite outboxes ---

    def download_events_message_with_new_connection(
        self, file_id: str, peer_email: str
    ) -> FileChangeEventsMessage | None:
        """Download from outbox using a new connection (thread-safe).

        Returns None when the message was rejected, or when it cannot be
        read yet (see ``ConnectionRouter.watcher_read_events_message``).
        """
        connection = self.connection_router.connection_for_parallel_download()
        return self.connection_router.watcher_read_events_message(
            connection, peer_email, file_id
        )

    def download_collection_file_with_new_connection(
        self, file_id: str, owner_email: str
    ) -> bytes:
        """Download collection file using a new connection (thread-safe)."""
        connection = self.connection_router.connection_for_parallel_download()
        data = connection.watcher_download_collection_file(file_id)
        data = self.connection_router.peer_store.decrypt_dataset_if_needed(
            owner_email, data
        )
        return data

    def sync_down(self, peer_emails: list[str]):
        for peer_email in peer_emails:
            # Sync messages with parallel download
            event_count = self.datasite_watcher_cache.sync_down_parallel(
                peer_email,
                self._executor,
                lambda fid,
                pe=peer_email: self.download_events_message_with_new_connection(
                    fid, pe
                ),
            )
            if event_count:
                print(f"Pulled {event_count} inbound event(s) from {peer_email}")
            # Sync collections with parallel download
            self.datasite_watcher_cache.sync_down_collections_parallel(
                peer_email,
                self._executor,
                lambda fid,
                pe=peer_email: self.download_collection_file_with_new_connection(
                    fid, pe
                ),
            )
