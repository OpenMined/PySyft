from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List

from pydantic import BaseModel, Field, PrivateAttr

from syft_client.sync.connections.base_connection import ConnectionConfig
from syft_client.sync.connections.connection_router import ConnectionRouter
from syft_client.sync.callback_mixin import BaseModelCallbackMixin
from syft_client.sync.events.file_change_event import FileChangeEventsMessage
from syft_client.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
)
from syft_client.sync.sync.caches.datasite_watcher_cache import (
    DataSiteWatcherCache,
    DataSiteWatcherCacheConfig,
)


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
        self, relative_path: Path, content: str, datasite_email: str | None = None
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
        file_changes = []
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

            file_change = self.get_proposed_file_change_object(
                path_in_datasite, content, datasite_email=recipient
            )
            file_changes.append(file_change)

        message = ProposedFileChangesMessage(
            sender_email=self.email, proposed_file_changes=file_changes
        )
        self.connection_router.watcher_send_proposed_file_changes_message(
            recipient, message
        )

        # Update local cache with new hashes
        for fc in file_changes:
            path_key = Path(fc.datasite_email) / fc.path_in_datasite
            if fc.is_deleted:
                self.datasite_watcher_cache.file_hashes.pop(path_key, None)
            else:
                self.datasite_watcher_cache.file_hashes[path_key] = fc.new_hash

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
    ) -> FileChangeEventsMessage:
        """Download from outbox using a new connection (thread-safe)."""
        connection = self.connection_router.connection_for_parallel_download()
        raw = connection.download_file(file_id)
        raw = self.connection_router.peer_store.decrypt_and_verify_if_needed(
            peer_email, raw
        )
        return FileChangeEventsMessage.from_compressed_data(raw)

    def download_dataset_file_with_new_connection(
        self, file_id: str, owner_email: str
    ) -> bytes:
        """Download dataset file using a new connection (thread-safe)."""
        connection = self.connection_router.connection_for_parallel_download()
        data = connection.watcher_download_dataset_file(file_id)
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
            # Sync datasets with parallel download
            self.datasite_watcher_cache.sync_down_datasets_parallel(
                peer_email,
                self._executor,
                lambda fid,
                pe=peer_email: self.download_dataset_file_with_new_connection(fid, pe),
            )
