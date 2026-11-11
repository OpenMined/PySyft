import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from syft.sync.sync.send_queue import SendQueue, datasite_send_queue_path


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
    send_queue: SendQueue

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
            send_queue=SendQueue(
                datasite_send_queue_path(config.syftbox_folder, config.email)
            ),
        )
        syncer._record_queued_hashes()
        return syncer

    # --- Pushing proposed file changes ---

    def _record_queued_hashes(self) -> None:
        """Record the hashes of the saved messages, so that a change after a
        restart builds on a saved change that is not sent yet. The content of
        the files is not read."""
        for recipient in self.send_queue.recipients():
            for msg_id in self.send_queue.msg_ids(recipient):
                entry = self.send_queue.read_entry(recipient, msg_id)
                self._record_proposed_hashes(entry.message)

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

    def push_file_changes(self, files: list[tuple[Path, str | bytes | None]]) -> None:
        """Save the changes to ``files`` as one message per datasite, then
        upload the saved messages to those datasites. A message that an upload
        error stops stays saved, and so do the messages after it.

        Each of ``files`` is a path in the syftbox and its content. A content of
        None is read from the file on disk.

        Raises:
            SyftPeerNotReadyError: when a recipient cannot receive yet.
                ``sync_up`` sends its message later. The other recipients still
                get their messages.
            ConnectionUnavailableError: when the backend cannot be reached.
                ``sync_up`` sends the messages later.
        """
        recipients = self._queue_file_changes(files)
        try:
            not_ready_errors = self._upload_queued(recipients)
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

    def _queue_file_changes(
        self, files: list[tuple[Path, str | bytes | None]]
    ) -> list[str]:
        """Save the changes to ``files`` in the send queue, as one message per
        datasite, and return those datasites. The content of the files is not
        held in memory after this returns, so the upload reads it back one
        message at a time."""
        # One message per datasite: each change goes to the owner of its path.
        changes_by_recipient: Dict[str, List[ProposedFileChange]] = {}
        for relative_path, content in files:
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
            self.send_queue.add(recipient, message)
            # The next change to a path builds on this one, sent or not.
            self._record_proposed_hashes(message)
        return list(changes_by_recipient)

    def _record_proposed_hashes(self, message: ProposedFileChangesMessage) -> None:
        hashes = self.datasite_watcher_cache.file_hashes
        for fc in message.proposed_file_changes:
            path_key = Path(fc.datasite_email) / fc.path_in_datasite
            if fc.is_deleted:
                hashes.pop(path_key, None)
            else:
                hashes[path_key] = fc.new_hash

    def _upload_queued(
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
        with self.send_queue.locked():
            for recipient in recipients:
                try:
                    self._upload_queued_to(recipient)
                except SyftPeerNotReadyError as e:
                    not_ready_errors[recipient] = e
        return not_ready_errors

    def _upload_queued_to(self, recipient: str) -> None:
        """Send the saved messages to ``recipient``, one message in memory at
        a time."""
        for msg_id in self.send_queue.msg_ids(recipient):
            message = self.send_queue.read_message(recipient, msg_id)
            # The DO refuses a copy that an upload stored before a stop.
            self.connection_router.watcher_send_proposed_file_changes_message(
                recipient, message
            )
            self.send_queue.remove(recipient, msg_id)

    def sync_up(self) -> None:
        """Upload every saved message. Warn for each recipient that cannot
        receive yet; its messages wait for the next sync."""
        not_ready_errors = self._upload_queued(self.send_queue.recipients())
        for recipient, error in not_ready_errors.items():
            warnings.warn(
                f"Changes to {recipient} wait in "
                f"{self.send_queue.recipient_folder(recipient)}: {error}"
            )

    def on_file_change(
        self, relative_path: Path | str, content: str | bytes | None = None
    ) -> None:
        """Push the change to one file; see ``push_file_changes``."""
        self.push_file_changes([(Path(relative_path), content)])

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
