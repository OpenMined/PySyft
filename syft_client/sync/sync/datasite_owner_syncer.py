import logging
from pathlib import Path
from uuid import uuid4

from pydantic import ConfigDict, Field, BaseModel, PrivateAttr
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Tuple
from syft_client.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
    FileChangeEvent,
)
from syft_client.sync.connections.base_connection import (
    ConnectionConfig,
    FileCollection,
)
from syft_client.sync.sync.caches.datasite_owner_cache import (
    DataSiteOwnerEventCacheConfig,
)
from syft_client.sync.connections.connection_router import ConnectionRouter
from syft_client.sync.sync.caches.datasite_owner_cache import DataSiteOwnerEventCache
from syft_client.sync.callback_mixin import BaseModelCallbackMixin
from syft_client.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft_client.sync.utils.path_filters import is_normal_syncable_path
from syft_client.sync.checkpoints.checkpoint import (
    Checkpoint,
    CheckpointFile,
    IncrementalCheckpoint,
    compact_incremental_checkpoints,
    DEFAULT_COMPACTING_THRESHOLD,
)
from syft_client.sync.checkpoints.rolling_state import RollingState
from syft_perms import SyftPermContext
from syft_client.sync.sync.constants import CACHE_DIR, ROLLING_STATE_FILENAME

logger = logging.getLogger(__name__)

# Default threshold for creating incremental checkpoint from rolling state
DEFAULT_CHECKPOINT_EVENT_THRESHOLD = 50

# Default: upload rolling state to GDrive after this many events
DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD = 1

# Minimum number of message files in a peer's outbox before
# compact_outbox_if_needed() will merge them into a single message.
MIN_MESSAGES_COMPACT = 20


class DatasiteOwnerSyncerConfig(BaseModel):
    email: str
    syftbox_folder: Path
    write_files: bool = True
    # Full path to collections folder - must be provided explicitly
    collections_folder: Path | None = None
    cache_config: DataSiteOwnerEventCacheConfig = Field(
        default_factory=DataSiteOwnerEventCacheConfig
    )
    connection_configs: List[ConnectionConfig] = []


class DatasiteOwnerSyncer(BaseModelCallbackMixin):
    """Responsible for downloading files and checking permissions"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    event_cache: DataSiteOwnerEventCache = Field(
        default_factory=lambda: DataSiteOwnerEventCache()
    )
    write_files: bool = True
    connection_router: ConnectionRouter
    initial_sync_done: bool = False
    email: str
    syftbox_folder: Path
    perm_context: SyftPermContext
    # Full path to collections folder
    collections_folder: Path | None = None

    syftbox_events_queue: Queue[FileChangeEventsMessage] = Field(default_factory=Queue)
    outbox_queue: Queue[Tuple[str, FileChangeEventsMessage]] = Field(
        default_factory=Queue
    )

    _executor: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=10)
    )
    # Cache of datasets shared with "any" - list of (tag, content_hash) tuples
    _any_shared_datasets: List[tuple] = PrivateAttr(default_factory=list)
    # Cache of read permissions per file path → frozenset of peer emails
    _read_perm_cache: dict[str, frozenset[str]] = PrivateAttr(default_factory=dict)

    # In-memory rolling state for tracking events since last checkpoint
    _rolling_state: RollingState | None = PrivateAttr(default=None)
    # Counter for events since last rolling state upload
    _events_since_rolling_state_upload: int = PrivateAttr(default=0)

    @classmethod
    def from_config(cls, config: DatasiteOwnerSyncerConfig):
        # Ensure cache config has the same collections_folder (both are now full paths)
        if config.collections_folder is not None:
            config.cache_config.collections_folder = config.collections_folder

        datasite = config.syftbox_folder / config.email
        datasite.mkdir(parents=True, exist_ok=True)

        return cls(
            event_cache=DataSiteOwnerEventCache.from_config(config.cache_config),
            write_files=config.write_files,
            connection_router=ConnectionRouter.from_configs(
                config.email, config.connection_configs
            ),
            email=config.email,
            syftbox_folder=config.syftbox_folder,
            perm_context=SyftPermContext(datasite=datasite),
            collections_folder=config.collections_folder,
        )

    @property
    def _rolling_state_path(self) -> Path:
        return self.syftbox_folder / CACHE_DIR / ROLLING_STATE_FILENAME

    def _load_rolling_state(self) -> None:
        """Load rolling state from disk for cross-process consistency."""
        path = self._rolling_state_path
        if not path.exists():
            return
        try:
            self._rolling_state = RollingState.model_validate_json(path.read_text())
        except Exception:
            pass

    def _save_rolling_state(self) -> None:
        """Save rolling state to disk for cross-process consistency."""
        if self._rolling_state is None:
            return
        path = self._rolling_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self._rolling_state.model_dump_json())
        tmp.rename(path)

    def sync(self, peer_emails: list[str], recompute_hashes: bool = True):
        self._load_rolling_state()
        try:
            if not self.initial_sync_done:
                self.pull_initial_state()

            if recompute_hashes:
                self.process_local_changes(recipients=peer_emails)

            # first, pull existing state
            for peer_email in peer_emails:
                proposed_count = 0
                while True:
                    msg = self.pull_and_process_next_proposed_filechange(
                        peer_email, raise_on_none=False
                    )
                    if msg is None:
                        # no new message, we are done
                        break
                    proposed_count += len(msg.proposed_file_changes)
                if proposed_count:
                    print(
                        f"Received {proposed_count} proposed change(s) "
                        f"from {peer_email}"
                    )
                self.process_syftbox_events_queue()
        finally:
            self._save_rolling_state()

    def download_events_message_by_id_with_connection(
        self, events_message_id: str
    ) -> FileChangeEventsMessage:
        # we need a new connection object because gdrive connections are not thread safe
        connection = self.connection_router.connection_for_eventlog(create_new=True)
        raw = connection.owner_download_raw_bytes_by_id(events_message_id)
        raw = self.connection_router.peer_store.decrypt_and_verify_for_self_if_needed(
            raw
        )
        return FileChangeEventsMessage.from_compressed_data(raw)

    def get_all_accepted_events_messages(
        self, since_timestamp: float | None = None
    ) -> list[FileChangeEventsMessage]:
        message_ids = self.connection_router.owner_get_all_accepted_event_file_ids(
            since_timestamp=since_timestamp
        )
        result_messages = self._executor.map(
            self.download_events_message_by_id_with_connection, message_ids
        )
        return list(result_messages)

    def pull_initial_state(self):
        """
        Pull initial state from Google Drive.

        Flow:
        1. Check for full (compacted) checkpoint → apply it
        2. Check for incremental checkpoints → apply them in order
        3. Check for rolling state → apply if valid
        4. Download any remaining events since last timestamp
        5. Ensure events_messages_connection is populated for get_cached_events()
        """
        events_since_timestamp: float | None = None
        restored_events: list[FileChangeEvent] = []

        # Step 1: Check for full (compacted) checkpoint
        full_checkpoint = self.connection_router.get_latest_checkpoint()
        if full_checkpoint is not None:
            print(
                f"Found full checkpoint with {len(full_checkpoint.files)} files, "
                "restoring..."
            )
            self.event_cache.apply_checkpoint(
                full_checkpoint, write_files=self.write_files
            )
            events_since_timestamp = full_checkpoint.last_event_timestamp

        # Step 2: Check for incremental checkpoints
        incremental_cps = self.connection_router.get_all_incremental_checkpoints()
        if incremental_cps:
            print(f"Found {len(incremental_cps)} incremental checkpoints, applying...")
            for inc_cp in incremental_cps:
                self._apply_incremental_checkpoint_to_cache(inc_cp)
                restored_events.extend(inc_cp.events)
                # Update timestamp to the latest event in this checkpoint
                for event in inc_cp.events:
                    if event.timestamp is not None:
                        if (
                            events_since_timestamp is None
                            or event.timestamp > events_since_timestamp
                        ):
                            events_since_timestamp = event.timestamp

        # Step 3: Check for rolling state
        rolling_state = self.connection_router.get_rolling_state()
        if rolling_state is not None and rolling_state.event_count > 0:
            print(
                f"Found rolling state with {rolling_state.event_count} events, "
                "applying..."
            )
            self._apply_rolling_state_to_cache(rolling_state)
            self._rolling_state = rolling_state
            restored_events.extend(rolling_state.events)

            # Update timestamp from rolling state
            if rolling_state.last_event_timestamp is not None:
                if (
                    events_since_timestamp is None
                    or rolling_state.last_event_timestamp > events_since_timestamp
                ):
                    events_since_timestamp = rolling_state.last_event_timestamp
        else:
            # Initialize empty rolling state
            base_timestamp = events_since_timestamp or 0.0
            self._rolling_state = RollingState(
                email=self.email,
                base_checkpoint_timestamp=base_timestamp,
            )

        # Step 4: Download any remaining events since last timestamp
        if events_since_timestamp is not None:
            events_messages = (
                self.connection_router.get_events_messages_since_timestamp(
                    events_since_timestamp
                )
            )
            if events_messages:
                print(
                    f"Downloading {len(events_messages)} events since "
                    "checkpoint/rolling state..."
                )
                for events_message in events_messages:
                    self.event_cache.add_events_message_to_local_cache(events_message)
                    self._add_events_to_rolling_state(events_message)
        elif full_checkpoint is None and not incremental_cps:
            # No checkpoints at all - download all events (fallback)
            print("No checkpoints found, downloading all events...")
            since_timestamp = self.event_cache.latest_cached_timestamp
            events_messages_list: list[FileChangeEventsMessage] = (
                self.get_all_accepted_events_messages(since_timestamp=since_timestamp)
            )
            for events_message in events_messages_list:
                self.event_cache.add_events_message_to_local_cache(events_message)

        # Step 5: Ensure events from checkpoints/rolling state are in
        # events_messages_connection. Steps 1-3 only populate file_hashes
        # and file_connection but not events_messages_connection, which
        # get_cached_events() reads from.
        if restored_events and not self.event_cache.get_cached_events():
            self._write_events_to_messages_cache(restored_events)

        # Load datasets from connection and populate _any_shared_datasets cache
        self._pull_datasets_for_initial_sync()

        # Restore private datasets from GDrive (owner-only collections)
        self._pull_private_datasets_for_initial_sync()

        self.initial_sync_done = True

    def _apply_incremental_checkpoint_to_cache(
        self, checkpoint: IncrementalCheckpoint
    ) -> None:
        """Apply events from an incremental checkpoint to the cache."""
        for event in checkpoint.events:
            if event.is_deleted:
                if event.path_in_datasite in self.event_cache.file_hashes:
                    del self.event_cache.file_hashes[event.path_in_datasite]
                if self.write_files:
                    self.event_cache.file_connection.delete_file(
                        str(event.path_in_datasite)
                    )
            else:
                self.event_cache.file_hashes[event.path_in_datasite] = event.new_hash
                if self.write_files:
                    self.event_cache.file_connection.write_file(
                        str(event.path_in_datasite), event.content
                    )

    def _pull_datasets_for_initial_sync(self):
        """Load datasets from GDrive when DO connects.

        Restores datasets to local filesystem and populates the _any_shared_datasets cache.
        """
        collections = (
            self.connection_router.owner_list_all_dataset_collections_with_permissions()
        )

        self._update_any_shared_datasets_cache(collections)

        collections_to_download = self._filter_collections_needing_download(collections)
        self._download_dataset_collections_parallel(collections_to_download)

    def _update_any_shared_datasets_cache(self, collections: list[FileCollection]):
        """Populate _any_shared_datasets cache from collections with 'any' permission."""
        for collection in collections:
            if collection.has_any_permission:
                entry = (collection.tag, collection.content_hash)
                if entry not in self._any_shared_datasets:
                    self._any_shared_datasets.append(entry)

    def _filter_collections_needing_download(
        self, collections: list[FileCollection]
    ) -> list[FileCollection]:
        """Return collections that don't exist locally or have different content hash."""
        from syft_client.sync.file_utils import compute_directory_hash

        result = []
        for collection in collections:
            # Use cached hash from event_cache first
            cached_hash = self.event_cache.get_collection_hash(collection.tag)
            if cached_hash is None and self.collections_folder is not None:
                # Fallback: compute hash from local filesystem (for locally created datasets)
                local_dataset_dir = self.collections_folder / collection.tag
                cached_hash = compute_directory_hash(local_dataset_dir)
                # Update cache if we computed a hash
                if cached_hash is not None:
                    self.event_cache.set_collection_hash(collection.tag, cached_hash)

            if cached_hash != collection.content_hash:
                result.append(collection)
        return result

    def _download_dataset_collections_parallel(self, collections: list[FileCollection]):
        """Download all files from collections in parallel and write to disk."""
        if not collections:
            return

        # Fetch file metadatas for all collections in parallel
        all_file_metadatas = list(
            self._executor.map(
                self._get_file_metadatas_with_new_connection, collections
            )
        )

        # Build list of (collection, file_metadata) tuples
        all_downloads = [
            (collection, metadata)
            for collection, file_metadatas in zip(collections, all_file_metadatas)
            for metadata in file_metadatas
        ]

        if not all_downloads:
            return

        # Download all files in parallel
        file_ids = [metadata["file_id"] for _, metadata in all_downloads]
        downloaded_contents = list(
            self._executor.map(self._download_file_with_new_connection, file_ids)
        )

        # Write all files to disk
        for (collection, metadata), content in zip(all_downloads, downloaded_contents):
            local_dataset_dir = self.collections_folder / collection.tag
            local_dataset_dir.mkdir(parents=True, exist_ok=True)
            (local_dataset_dir / metadata["file_name"]).write_bytes(content)

        # Update cached hashes for downloaded collections
        for collection in collections:
            self.event_cache.set_collection_hash(
                collection.tag, collection.content_hash
            )

    def _get_file_metadatas_with_new_connection(
        self, collection: FileCollection
    ) -> list:
        """Get file metadatas for a collection using a new connection for thread safety."""
        connection = self.connection_router.connection_for_parallel_download()
        return connection.watcher_get_dataset_collection_file_metadatas(
            tag=collection.tag,
            content_hash=collection.content_hash,
            owner_email=self.email,
        )

    def _download_file_with_new_connection(self, file_id: str) -> bytes:
        """Download a file using a new connection for thread safety."""
        connection = self.connection_router.connection_for_parallel_download()
        return connection.watcher_download_dataset_file(file_id)

    # =========================================================================
    # PRIVATE DATASET RESTORE METHODS
    # =========================================================================

    def _pull_private_datasets_for_initial_sync(self):
        """Restore private datasets from GDrive when DO reconnects.

        Downloads private data from owner-only collection folders
        to {syftbox_folder}/private/syft_datasets/{tag}/.
        """
        collections = self.connection_router.owner_list_private_dataset_collections()
        if not collections:
            return

        collections_to_download = self._filter_private_collections_needing_download(
            collections
        )
        self._download_private_collections_parallel(collections_to_download)

    def _private_dataset_local_dir(self, tag: str) -> Path:
        return self.syftbox_folder / self.email / "private" / "syft_datasets" / tag

    def _filter_private_collections_needing_download(
        self, collections: list[FileCollection]
    ) -> list[FileCollection]:
        """Return private collections that don't exist locally yet."""
        result = []
        for collection in collections:
            local_dir = self._private_dataset_local_dir(collection.tag)
            if not local_dir.exists() or not any(local_dir.iterdir()):
                result.append(collection)
        return result

    def _download_private_collections_parallel(self, collections: list[FileCollection]):
        """Download private collection files in parallel and write to disk."""
        if not collections:
            return

        all_file_metadatas = list(
            self._executor.map(
                self._get_private_file_metadatas_with_new_connection, collections
            )
        )

        all_downloads = [
            (collection, metadata)
            for collection, file_metadatas in zip(collections, all_file_metadatas)
            for metadata in file_metadatas
        ]

        if not all_downloads:
            return

        file_ids = [metadata["file_id"] for _, metadata in all_downloads]
        downloaded_contents = list(
            self._executor.map(self._download_file_with_new_connection, file_ids)
        )

        for (collection, metadata), content in zip(all_downloads, downloaded_contents):
            local_dir = self._private_dataset_local_dir(collection.tag)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / metadata["file_name"]).write_bytes(content)

        # Fix data_dir in private_metadata.yaml to point to current local path
        for collection in collections:
            self._fix_private_metadata_data_dir(collection.tag)

    def _get_private_file_metadatas_with_new_connection(
        self, collection: FileCollection
    ) -> list:
        """Get file metadatas for a private collection using a new connection."""
        connection = self.connection_router.connection_for_parallel_download()
        return connection.owner_get_private_collection_file_metadatas(
            tag=collection.tag,
            content_hash=collection.content_hash,
            owner_email=self.email,
        )

    def _fix_private_metadata_data_dir(self, dataset_tag: str):
        """Update data_dir in private_metadata.yaml to match the current syftbox path."""
        local_dir = self._private_dataset_local_dir(dataset_tag)
        metadata_path = local_dir / "private_metadata.yaml"
        if not metadata_path.exists():
            return

        import yaml

        data = yaml.safe_load(metadata_path.read_text())
        expected_dir = str(local_dir)
        if data.get("data_dir") != expected_dir:
            data["data_dir"] = expected_dir
            metadata_path.write_text(yaml.safe_dump(data, indent=2, sort_keys=False))

    def _get_readers(self, path: str, recipients: list[str]) -> frozenset[str]:
        """Return the set of recipients that have read access to the given path."""
        return frozenset(r for r in recipients if self.check_read_permissions(r, path))

    def _get_paths_under_perm_file(self, perm_path: str) -> list[str]:
        """Find all files under the perm file's parent directory on the filesystem."""
        datasite = self.syftbox_folder / self.email
        parent_dir = datasite / str(Path(perm_path).parent)
        if not parent_dir.exists():
            return []
        collections_rel = self.event_cache.collections_relative_path()
        paths = []
        for file_path in parent_dir.rglob("*"):
            if file_path.is_file():
                rel = str(file_path.relative_to(datasite))
                if is_normal_syncable_path(rel, collections_path=collections_rel):
                    paths.append(rel)
        return paths

    def _create_resend_event(self, path: str) -> "FileChangeEvent | None":
        """Create a FileChangeEvent for an existing file to resend to new readers."""
        try:
            content = self.event_cache.file_connection.read_file(path)
        except Exception:
            return None
        if content is None:
            return None
        from syft_client.sync.utils.syftbox_utils import (
            get_event_hash_from_content,
            create_event_timestamp,
        )

        timestamp = create_event_timestamp()
        return FileChangeEvent(
            id=uuid4(),
            path_in_datasite=Path(path),
            content=content,
            old_hash=None,
            new_hash=get_event_hash_from_content(content),
            submitted_timestamp=timestamp,
            timestamp=timestamp,
            datasite_email=self.email,
            is_deleted=False,
        )

    def process_local_changes(self, recipients: list[str]):
        file_change_events_message = self.event_cache.process_local_file_changes()
        if file_change_events_message is None:
            return

        self.syftbox_events_queue.put(file_change_events_message)

        perm_events, data_events = self._split_events(file_change_events_message.events)
        events_by_recipient: dict[str, list[FileChangeEvent]] = {
            r: [] for r in recipients
        }
        data_event_sent_to: dict[str, set[str]] = {}

        self._route_data_events(
            data_events, recipients, events_by_recipient, data_event_sent_to
        )
        self._route_perm_events(
            perm_events, recipients, events_by_recipient, data_event_sent_to
        )
        self._queue_events_for_outbox(events_by_recipient)
        # Feed local events into rolling state so they flow through
        # incremental/full checkpoints.
        self._add_events_to_rolling_state(file_change_events_message)
        self.process_syftbox_events_queue()

    def _split_events(
        self, events: list[FileChangeEvent]
    ) -> tuple[list[FileChangeEvent], list[FileChangeEvent]]:
        """Split events into (permission_events, data_events)."""
        from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

        perm_events = []
        data_events = []
        for event in events:
            if str(event.path_in_datasite).endswith(PERMISSION_FILE_NAME):
                perm_events.append(event)
            else:
                data_events.append(event)
        return perm_events, data_events

    def _route_data_events(
        self,
        data_events: list[FileChangeEvent],
        recipients: list[str],
        events_by_recipient: dict[str, list[FileChangeEvent]],
        data_event_sent_to: dict[str, set[str]],
    ):
        """Route data events to recipients who have read access."""
        for event in data_events:
            path_str = str(event.path_in_datasite)
            readers = self._get_readers(path_str, recipients)
            self._read_perm_cache[path_str] = readers
            data_event_sent_to[path_str] = set(readers)
            for reader in readers:
                events_by_recipient[reader].append(event)

    def _route_perm_events(
        self,
        perm_events: list[FileChangeEvent],
        recipients: list[str],
        events_by_recipient: dict[str, list[FileChangeEvent]],
        data_event_sent_to: dict[str, set[str]],
    ):
        """Share perm files with readers and resend affected files to newly-permitted readers."""
        for event in perm_events:
            path_str = str(event.path_in_datasite)
            perm_readers = self._get_readers(path_str, recipients)
            for reader in perm_readers:
                events_by_recipient[reader].append(event)

            affected_paths = self._get_paths_under_perm_file(path_str)
            for affected_path in affected_paths:
                new_readers = self._get_readers(affected_path, recipients)
                old_readers = self._read_perm_cache.get(affected_path, frozenset())
                newly_permitted = new_readers - old_readers
                self._read_perm_cache[affected_path] = new_readers

                if not newly_permitted:
                    continue
                already_sent = data_event_sent_to.get(affected_path, set())
                needs_resend = newly_permitted - already_sent
                if needs_resend:
                    resend_event = self._create_resend_event(affected_path)
                    if resend_event is not None:
                        for reader in needs_resend:
                            events_by_recipient[reader].append(resend_event)

    def _queue_events_for_outbox(
        self, events_by_recipient: dict[str, list[FileChangeEvent]]
    ):
        """Wrap events into messages per recipient and queue for outbox."""
        for recipient, events in events_by_recipient.items():
            if events:
                msg = FileChangeEventsMessage(events=events)
                self.outbox_queue.put((recipient, msg))

    def pull_and_process_next_proposed_filechange(
        self, sender_email: str, raise_on_none=True
    ) -> ProposedFileChangesMessage | None:
        # raise on none is useful for testing, shouldnt be used in production
        message = self.connection_router.owner_get_next_proposed_filechange_message(
            sender_email=sender_email
        )
        if message is not None:
            sender_email = message.sender_email
            self.handle_proposed_filechange_events_message(sender_email, message)

            # delete the message once we are done
            self.connection_router.owner_remove_proposed_filechange_from_inbox(message)
            return message
        elif raise_on_none:
            raise ValueError("No proposed file change to process")
        else:
            return None

    def check_write_permission(self, sender_email: str, path: str) -> bool:
        """Check if sender has write access to the given path."""
        self.perm_context._reload()
        return self.perm_context.open(path).has_write_access(sender_email)

    def check_read_permissions(self, recipient_email: str, path: str) -> bool:
        """Check if recipient has read access to the given path."""
        self.perm_context._reload()
        return self.perm_context.open(path).has_read_access(recipient_email)

    def handle_proposed_filechange_events_message(
        self, sender_email: str, proposed_events_message: ProposedFileChangesMessage
    ):
        collections_rel = self.event_cache.collections_relative_path()
        allowed_changes = [
            change
            for change in proposed_events_message.proposed_file_changes
            if is_normal_syncable_path(
                str(change.path_in_datasite), collections_path=collections_rel
            )
            and self.check_write_permission(sender_email, str(change.path_in_datasite))
        ]

        if not allowed_changes:
            return

        filtered_message = ProposedFileChangesMessage(
            sender_email=proposed_events_message.sender_email,
            proposed_file_changes=allowed_changes,
        )

        accepted_events_message = self.event_cache.process_proposed_events_message(
            filtered_message
        )
        if accepted_events_message is not None:
            self.queue_event_for_syftbox(
                recipients=[sender_email],
                file_change_events_message=accepted_events_message,
            )
            # Add to rolling state after processing
            self._add_events_to_rolling_state(accepted_events_message)

    def queue_event_for_syftbox(
        self, recipients: list[str], file_change_events_message: FileChangeEventsMessage
    ):
        self.syftbox_events_queue.put(file_change_events_message)

        for recipient in recipients:
            self.outbox_queue.put((recipient, file_change_events_message))

    def process_syftbox_events_queue(self):
        # TODO: make this atomic
        while not self.syftbox_events_queue.empty():
            file_change_events_message = self.syftbox_events_queue.get()
            self.connection_router.owner_write_events_message_to_syftbox(
                file_change_events_message
            )
        while not self.outbox_queue.empty():
            recipient, file_change_events_message = self.outbox_queue.get()
            self.connection_router.owner_write_event_messages_to_outbox(
                recipient, file_change_events_message
            )

    def _download_outbox_message_with_new_connection(
        self, file_id: str
    ) -> FileChangeEventsMessage:
        """Download and decompress one outbox events-message file."""
        connection = self.connection_router.connection_for_parallel_download()
        raw = connection.download_file(file_id)
        return FileChangeEventsMessage.from_compressed_data(raw)

    def compact_outbox_if_needed(
        self,
        recipient_email: str,
        min_messages: int = MIN_MESSAGES_COMPACT,
    ) -> int:
        """Merge accumulated event-message files in a recipient's outbox into one.

        Compaction downloads them, concatenates their events into one fresh message, and
        deletes the originals.

        Returns the number of source messages compacted (0 if below threshold,
        if no outbox folder, or if peer encryption is on — DO can't read
        peer-encrypted bytes back to merge).
        """
        if self.connection_router.peer_store.peer_uses_encryption(recipient_email):
            logger.debug(
                f"outbox compaction for {recipient_email} skipped — "
                f"peer encryption is not supported yet"
            )
            return 0

        conn = self.connection_router.connection_for_outbox()
        folder_id = conn._get_own_datasite_outbox_id(recipient_email)
        if folder_id is None:
            logger.info(
                f"WARNING: outbox compaction for {recipient_email} skipped — "
                f"could not resolve outbox folder id"
            )
            return 0

        metas = conn.get_file_metadatas_from_folder(folder_id, since_timestamp=None)

        # Keep only well-formed events-message filenames; track timestamps so
        # we apply events in order in the merged message.
        parsed: list[tuple[float, str]] = []
        for m in metas:
            try:
                fname = FileChangeEventsMessageFileName.from_string(m["name"])
            except ValueError:
                continue
            parsed.append((fname.timestamp, m["id"]))

        if len(parsed) < min_messages:
            return 0

        parsed.sort(key=lambda x: x[0])

        # Parallel download. executor.map preserves input order, so messages[i]
        # corresponds to parsed[i] — no re-zip-and-sort needed since `parsed`
        # is already timestamp-sorted.
        file_ids = [fid for _, fid in parsed]
        messages = list(
            self._executor.map(
                self._download_outbox_message_with_new_connection, file_ids
            )
        )

        merged_events: list[FileChangeEvent] = []
        for msg in messages:
            merged_events.extend(msg.events)

        merged = FileChangeEventsMessage(events=merged_events)

        # Write the merged message FIRST. A DS that pulls between this and
        # the delete sees `merged ∪ some originals`; its file_hashes cache
        # makes duplicate events no-op writes.
        self.connection_router.owner_write_event_messages_to_outbox(
            recipient_email, merged
        )
        conn.delete_multiple_files_by_ids(file_ids)
        return len(parsed)

    def write_file_filesystem(self, path: str, content: str):
        if self.write_files:
            raise NotImplementedError("Writing files to filesystem is not implemented")

    # =========================================================================
    # ROLLING STATE METHODS
    # =========================================================================

    def _write_events_to_messages_cache(self, events: list[FileChangeEvent]) -> None:
        """Write events to the events_messages_connection so get_cached_events() can find them."""
        events_message = FileChangeEventsMessage(events=events)
        self.event_cache.events_messages_connection.write_file(
            path=events_message.message_filepath.as_string(),
            content=events_message,
        )

    def _apply_rolling_state_to_cache(self, rolling_state: RollingState) -> None:
        """Apply events from rolling state to the cache."""
        for event in rolling_state.events:
            if event.is_deleted:
                if event.path_in_datasite in self.event_cache.file_hashes:
                    del self.event_cache.file_hashes[event.path_in_datasite]
                if self.write_files:
                    self.event_cache.file_connection.delete_file(
                        str(event.path_in_datasite)
                    )
            else:
                self.event_cache.file_hashes[event.path_in_datasite] = event.new_hash
                if self.write_files:
                    self.event_cache.file_connection.write_file(
                        str(event.path_in_datasite), event.content
                    )

    def _add_events_to_rolling_state(
        self,
        events_message: FileChangeEventsMessage,
        upload_threshold: int = DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD,
    ) -> None:
        """
        Add events to the in-memory rolling state and upload if threshold is reached.

        Args:
            events_message: The events to add.
            upload_threshold: Upload to GDrive after this many events added.
        """
        if self._rolling_state is None:
            return

        self._rolling_state.add_events_message(events_message)
        self._events_since_rolling_state_upload += len(events_message.events)

        # Upload if threshold reached
        if self._events_since_rolling_state_upload >= upload_threshold:
            self._upload_rolling_state()

    def _upload_rolling_state(self) -> None:
        """Upload the in-memory rolling state to GDrive."""
        if self._rolling_state is None or self._rolling_state.event_count == 0:
            return

        self.connection_router.upload_rolling_state(self._rolling_state)
        self._events_since_rolling_state_upload = 0

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def create_incremental_checkpoint(self) -> IncrementalCheckpoint:
        """
        Create an incremental checkpoint from the current rolling state.

        Deduplicates events by path (keeps only latest version per file),
        then converts to an incremental checkpoint, uploads it,
        and deletes the rolling state.

        Returns:
            The created IncrementalCheckpoint object.
        """
        if self._rolling_state is None or self._rolling_state.event_count == 0:
            raise ValueError("No rolling state to create checkpoint from")

        # Get the next sequence number
        seq_number = self.connection_router.get_next_incremental_sequence_number()

        # Deduplicate events: keep only latest version per file path
        # This reduces storage when the same file changes multiple times
        latest_events: dict[str, "FileChangeEvent"] = {}
        for event in self._rolling_state.events:
            path_key = str(event.path_in_datasite)
            latest_events[path_key] = event

        # Create incremental checkpoint with deduplicated events
        deduplicated_events = list(latest_events.values())
        incremental_cp = IncrementalCheckpoint(
            email=self.email,
            sequence_number=seq_number,
            events=deduplicated_events,
        )

        # Upload to Google Drive
        original_count = self._rolling_state.event_count
        deduplicated_count = len(deduplicated_events)
        print(
            f"Creating incremental checkpoint #{seq_number} "
            f"with {deduplicated_count} events "
            f"(deduplicated from {original_count})..."
        )
        self.connection_router.upload_incremental_checkpoint(incremental_cp)

        # Delete rolling state from GDrive and reset in-memory state
        self.connection_router.delete_rolling_state()
        base_timestamp = (
            self._rolling_state.last_event_timestamp or incremental_cp.timestamp
        )
        self._rolling_state = RollingState(
            email=self.email,
            base_checkpoint_timestamp=base_timestamp,
        )
        self._events_since_rolling_state_upload = 0

        print("Incremental checkpoint created, rolling state reset!")

        return incremental_cp

    def create_checkpoint(self) -> Checkpoint:
        """
        Create a full checkpoint from current cache state.

        This creates a full checkpoint and resets the rolling state.
        Used for legacy compatibility and when manually creating checkpoints.

        Returns:
            The created Checkpoint object.
        """
        self._load_rolling_state()
        try:
            last_event_timestamp = self.event_cache.get_latest_event_timestamp()
            checkpoint = self.event_cache.create_checkpoint(
                last_event_timestamp=last_event_timestamp
            )
            print(f"Creating full checkpoint with {len(checkpoint.files)} files...")
            self.connection_router.upload_checkpoint(checkpoint)

            # Reset rolling state after checkpoint creation
            self.connection_router.delete_rolling_state()
            base_timestamp = last_event_timestamp or checkpoint.timestamp
            self._rolling_state = RollingState(
                email=self.email,
                base_checkpoint_timestamp=base_timestamp,
            )
            self._events_since_rolling_state_upload = 0

            return checkpoint
        finally:
            self._save_rolling_state()

    def should_create_checkpoint(
        self, threshold: int = DEFAULT_CHECKPOINT_EVENT_THRESHOLD
    ) -> bool:
        """
        Check if we should create an incremental checkpoint.

        Based on rolling state event count (not GDrive event files).

        Args:
            threshold: Create checkpoint if rolling state has >= threshold events.

        Returns:
            True if checkpoint should be created.
        """
        if self._rolling_state is None:
            return False
        return self._rolling_state.event_count >= threshold

    def should_compact_checkpoints(
        self, threshold: int = DEFAULT_COMPACTING_THRESHOLD
    ) -> bool:
        """
        Check if we should compact incremental checkpoints into a full checkpoint.

        Args:
            threshold: Compact if number of incremental checkpoints >= threshold.

        Returns:
            True if compacting should happen.
        """
        count = self.connection_router.get_incremental_checkpoint_count()
        return count >= threshold

    def compact_checkpoints(self) -> Checkpoint:
        """
        Compact all incremental checkpoints into a single full checkpoint.

        If an existing full checkpoint exists, it is merged with all incremental
        checkpoints to create the new full checkpoint. This ensures no data loss.

        Downloads existing full checkpoint (if any) and all incremental checkpoints,
        merges them, uploads the compacted full checkpoint, and deletes all
        incremental checkpoints.

        Returns:
            The compacted Checkpoint object.
        """
        print("Compacting incremental checkpoints...")

        # Download existing full checkpoint (if exists)
        existing_checkpoint = self.connection_router.get_latest_checkpoint()

        # Download all incremental checkpoints
        incremental_cps = self.connection_router.get_all_incremental_checkpoints()

        if not incremental_cps:
            raise ValueError("No incremental checkpoints to compact")

        print(f"Found {len(incremental_cps)} incremental checkpoints to compact")

        # Start with existing checkpoint files (if any)
        if existing_checkpoint:
            print(
                f"Merging with existing full checkpoint "
                f"({len(existing_checkpoint.files)} files)..."
            )
            # Convert existing checkpoint files to events for merging
            merged_events: dict[str, FileChangeEvent] = {}

            # Add existing checkpoint files as events
            for checkpoint_file in existing_checkpoint.files:
                # Create a FileChangeEvent from CheckpointFile
                event = FileChangeEvent(
                    id=uuid4(),
                    path_in_datasite=Path(checkpoint_file.path),
                    datasite_email=self.email,
                    content=checkpoint_file.content,
                    old_hash=None,
                    new_hash=checkpoint_file.hash,
                    is_deleted=False,
                    submitted_timestamp=existing_checkpoint.timestamp,
                    timestamp=existing_checkpoint.timestamp,
                )
                merged_events[checkpoint_file.path] = event

            # Merge incremental checkpoints on top (later events overwrite)
            for inc_cp in sorted(incremental_cps, key=lambda c: c.sequence_number):
                for event in inc_cp.events:
                    merged_events[str(event.path_in_datasite)] = event

            # Create full checkpoint from merged events
            files = []
            for event in merged_events.values():
                if not event.is_deleted and event.content is not None:
                    files.append(
                        CheckpointFile(
                            path=str(event.path_in_datasite),
                            hash=str(event.new_hash),
                            content=event.content,
                        )
                    )

            # Find latest timestamp
            last_event_timestamp = existing_checkpoint.last_event_timestamp
            for inc_cp in incremental_cps:
                for event in inc_cp.events:
                    if event.timestamp is not None:
                        if (
                            last_event_timestamp is None
                            or event.timestamp > last_event_timestamp
                        ):
                            last_event_timestamp = event.timestamp

            compacted = Checkpoint(
                email=self.email,
                files=files,
                last_event_timestamp=last_event_timestamp,
            )
        else:
            # No existing checkpoint, just merge incremental checkpoints
            compacted = compact_incremental_checkpoints(incremental_cps, self.email)

        # Upload the compacted checkpoint (this deletes old full checkpoints)
        print(f"Uploading compacted checkpoint with {len(compacted.files)} files...")
        self.connection_router.upload_checkpoint(compacted)

        # Delete all incremental checkpoints
        self.connection_router.delete_all_incremental_checkpoints()

        print("Compacting complete!")

        return compacted

    def try_create_checkpoint(
        self,
        threshold: int = DEFAULT_CHECKPOINT_EVENT_THRESHOLD,
        compacting_threshold: int = DEFAULT_COMPACTING_THRESHOLD,
    ) -> IncrementalCheckpoint | Checkpoint | None:
        """
        Try to create incremental checkpoint and/or compact if thresholds exceeded.

        Args:
            threshold: Create incremental checkpoint if rolling state has >= events.
            compacting_threshold: Compact if >= this many incremental checkpoints.

        Returns:
            The created checkpoint (incremental or compacted), or None.
        """
        self._load_rolling_state()
        try:
            result = None

            # First, check if we should create an incremental checkpoint
            if self.should_create_checkpoint(threshold):
                result = self.create_incremental_checkpoint()

                # After creating, check if we should compact
                if self.should_compact_checkpoints(compacting_threshold):
                    result = self.compact_checkpoints()

            return result
        finally:
            self._save_rolling_state()
