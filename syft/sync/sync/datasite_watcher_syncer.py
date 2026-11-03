import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, List

from pydantic import BaseModel, Field, PrivateAttr

from syft.sync.callback_mixin import BaseModelCallbackMixin
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
)
from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.sync.caches.datasite_watcher_cache import (
    DataSiteWatcherCache,
    DataSiteWatcherCacheConfig,
)
from syft.sync.sync.spool import Spool, datasite_spool_path


class DatasiteWatcherSyncerConfig(BaseModel):
    syftbox_folder: Path
    email: str
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
    spool: Spool
    queue: Queue = Field(default_factory=Queue)

    _executor: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=10)
    )

    @classmethod
    def from_config(
        cls, config: DatasiteWatcherSyncerConfig, connection_router: ConnectionRouter
    ):
        """``config`` holds plain values only. ``connection_router`` is live and
        shared with the other parts of the manager, so the caller builds it
        once and passes it in next to the config (see
        ``SyftboxManager.from_config``)."""
        syncer = cls(
            syftbox_folder=config.syftbox_folder,
            email=config.email,
            connection_router=connection_router,
            datasite_watcher_cache=DataSiteWatcherCache.from_config(
                config.datasite_watcher_cache_config, connection_router
            ),
            spool=Spool(datasite_spool_path(config.syftbox_folder, config.email)),
        )
        syncer._record_spooled_hashes()
        return syncer

    # --- Pushing proposed file changes ---

    def _record_spooled_hashes(self) -> None:
        """Record the hashes of the saved messages, so that a change after a
        restart builds on a saved change that is not sent yet."""
        for recipient in self.spool.recipients():
            for message in self.spool.pending_messages(recipient):
                self._record_proposed_hashes(message)

    def get_proposed_file_change_object(
        self,
        relative_path: Path,
        content: str | bytes,
        datasite_email: str | None = None,
    ) -> ProposedFileChange:
        if datasite_email is None:
            datasite_email = self.email
        cache_key = Path(datasite_email) / relative_path
        old_hash = self.datasite_watcher_cache.current_hash_for_file(cache_key)
        return ProposedFileChange(
            datasite_email=datasite_email,
            path_in_datasite=relative_path,
            content=content,
            old_hash=old_hash,
        )

    def process_file_changes_queue(self):
        """Save the queued changes as one message per datasite, then upload the
        saved messages to those datasites. A message that an upload error stops
        stays saved, and so do the messages after it.

        Raises:
            SyftPeerNotReadyError: when a recipient cannot receive yet.
                ``sync_up`` sends its message later. The other recipients still
                get their messages.
            ConnectionUnavailableError: when the backend cannot be reached.
                ``sync_up`` sends the messages later.
        """
        # One message per datasite: each change goes to the owner of its path.
        changes_by_recipient: Dict[str, List[ProposedFileChange]] = {}
        while not self.queue.empty():
            relative_path, content = self.queue.get()

            # for in memory connection we pass content directly
            if content is None:
                raw = (self.syftbox_folder / relative_path).read_bytes()
                # A text file goes as text, any other file as bytes.
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw

            datasite_email = relative_path.parts[0]
            path_in_datasite = (
                Path(*relative_path.parts[1:])
                if len(relative_path.parts) > 1
                else Path()
            )

            # TODO: add some better parsing logic here
            recipient = datasite_email
            path_in_datasite = path_in_datasite

            file_change = self.get_proposed_file_change_object(
                path_in_datasite, content, datasite_email=recipient
            )
            changes_by_recipient.setdefault(recipient, []).append(file_change)
        for recipient, changes in changes_by_recipient.items():
            message = ProposedFileChangesMessage(
                sender_email=self.email, proposed_file_changes=changes
            )
            self.spool.add(recipient, message)
            # The next change to a path builds on this one, sent or not.
            self._record_proposed_hashes(message)
        try:
            not_ready_errors = self._upload_spooled(changes_by_recipient)
        except ConnectionUnavailableError as e:
            raise ConnectionUnavailableError(
                f"{e} The change is saved, and client.sync() sends it."
            ) from e
        if not_ready_errors:
            error = next(iter(not_ready_errors.values()))
            raise SyftPeerNotReadyError(
                error.peer_email,
                cause=error.cause,
                remedy=f"{error.remedy} The change is saved, and client.sync() "
                "sends it once the peer is ready.",
            ) from error

    def _record_proposed_hashes(self, message: ProposedFileChangesMessage) -> None:
        hashes = self.datasite_watcher_cache.file_hashes
        for fc in message.proposed_file_changes:
            path_key = Path(fc.datasite_email) / fc.path_in_datasite
            if fc.is_deleted:
                hashes.pop(path_key, None)
            else:
                hashes[path_key] = fc.new_hash

    def _upload_spooled(
        self, recipients: Iterable[str]
    ) -> dict[str, SyftPeerNotReadyError]:
        """Upload the saved messages to each of ``recipients``, oldest first,
        and remove each message once its recipient has it.

        Returns why the messages to each recipient that cannot receive yet
        wait, by recipient email. They stay saved, in order.

        Raises:
            ConnectionUnavailableError: when the backend cannot be reached. An
                upload that fails for another reason raises too. That message
                and the later ones stay saved.
        """
        not_ready_errors = {}
        with self.spool.locked():
            for recipient in recipients:
                try:
                    self._upload_spooled_to(recipient)
                except SyftPeerNotReadyError as e:
                    not_ready_errors[recipient] = e
        return not_ready_errors

    def _upload_spooled_to(self, recipient: str) -> None:
        for message in self.spool.pending_messages(recipient):
            msg_id = message.message_filename.as_string()
            # An earlier upload may have stopped before it removed the message.
            if not self.connection_router.is_delivered(recipient, msg_id):
                self.connection_router.watcher_send_proposed_file_changes_message(
                    recipient, message
                )
            self.spool.remove(recipient, msg_id)

    def sync_up(self) -> None:
        """Upload every saved message. Warn for each recipient that cannot
        receive yet; its messages wait for the next sync."""
        not_ready_errors = self._upload_spooled(self.spool.recipients())
        for recipient, error in not_ready_errors.items():
            warnings.warn(
                f"Changes to {recipient} wait in "
                f"{self.spool.recipient_folder(recipient)}: {error}"
            )

    def on_file_change(
        self,
        relative_path: Path | str,
        content: str | bytes | None = None,
        process_now=True,
    ):
        relative_path = Path(relative_path)
        self.queue.put((relative_path, content))
        if process_now:
            self.process_file_changes_queue()

    # --- Pulling from datasite outboxes ---

    def download_collection_file_with_new_connection(
        self, file_id: str, owner_email: str
    ) -> bytes:
        """Download collection file using a new connection (thread-safe)."""
        collections = (
            self.connection_router.connections.collections_con.clone_for_thread()
        )
        data = collections.fetch_file(file_id)
        data = self.connection_router.peer_store.decrypt_dataset_if_needed(
            owner_email, data
        )
        return data

    def sync_down(self, peer_emails: list[str]):
        for peer_email in peer_emails:
            event_count = self.datasite_watcher_cache.sync_down(peer_email)
            if event_count:
                print(f"Pulled {event_count} inbound event(s) from {peer_email}")
            # Sync collections with parallel download
            self.datasite_watcher_cache.sync_down_collections_parallel(
                peer_email,
                self._executor,
                lambda fid, pe=peer_email: (
                    self.download_collection_file_with_new_connection(fid, pe)
                ),
            )
