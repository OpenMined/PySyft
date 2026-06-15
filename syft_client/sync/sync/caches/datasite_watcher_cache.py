from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List
from syft_client.sync.sync.caches.cache_file_writer_connection import FSFileConnection
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from syft_client.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
)
from syft_client.sync.connections.connection_router import ConnectionRouter
from syft_client.sync.connections.base_connection import ConnectionConfig
from syft_client.sync.sync.caches.cache_file_writer_connection import (
    CacheFileConnection,
    InMemoryCacheFileConnection,
)

SECONDS_BEFORE_SYNCING_DOWN = 0


class DataSiteWatcherCacheConfig(BaseModel):
    email: str = ""
    use_in_memory_cache: bool = True
    syftbox_folder: Path | None = None
    events_base_path: Path | None = None
    connection_configs: List[ConnectionConfig] = []
    # Subpath from owner_email to collections folder (e.g., "public/syft_datasets")
    collection_subpath: Path | None = None


class DataSiteWatcherCache(BaseModel):
    events_connection: CacheFileConnection = Field(
        default_factory=InMemoryCacheFileConnection
    )

    file_connection: CacheFileConnection = Field(
        default_factory=InMemoryCacheFileConnection
    )

    file_hashes: Dict[str, int] = {}
    current_check_point: str = None
    connection_router: ConnectionRouter
    last_sync: datetime | None = None
    seconds_before_syncing_down: int = SECONDS_BEFORE_SYNCING_DOWN
    peers: List[str] = []
    last_event_timestamp_per_peer: Dict[str, float] = {}
    # Base syftbox folder
    syftbox_folder: Path | None = None
    # Subpath from owner_email to collections folder (e.g., "public/syft_datasets")
    collection_subpath: Path | None = None
    # Cache of dataset collection hashes: path -> content_hash
    dataset_collection_hashes: Dict[Path, str] = {}

    @classmethod
    def from_config(cls, config: DataSiteWatcherCacheConfig):
        if config.use_in_memory_cache:
            res = cls(
                events_connection=InMemoryCacheFileConnection[FileChangeEvent](),
                file_connection=InMemoryCacheFileConnection[str](),
                connection_router=ConnectionRouter.from_configs(
                    email=config.email,
                    connection_configs=config.connection_configs,
                ),
                syftbox_folder=config.syftbox_folder,
                collection_subpath=config.collection_subpath,
            )
            return res
        else:
            if config.syftbox_folder is None:
                raise ValueError("syftbox_folder is required for non-in-memory cache")
            if config.collection_subpath is None:
                raise ValueError(
                    "collection_subpath is required for non-in-memory cache"
                )

            syftbox_folder_name = Path(config.syftbox_folder).name
            syftbox_parent = Path(config.syftbox_folder).parent
            events_folder = syftbox_parent / f"{syftbox_folder_name}-event-messages"

            cache = cls(
                events_connection=FSFileConnection(
                    base_dir=events_folder, dtype=FileChangeEventsMessage
                ),
                file_connection=FSFileConnection(base_dir=config.syftbox_folder),
                connection_router=ConnectionRouter.from_configs(
                    email=config.email,
                    connection_configs=config.connection_configs,
                ),
                syftbox_folder=config.syftbox_folder,
                collection_subpath=config.collection_subpath,
            )
            cache._load_cached_state()
            return cache

    def _load_cached_state(self):
        """Load cached state from disk: file hashes, timestamps, and dataset hashes."""
        self._load_file_hashes_from_events()
        self._load_dataset_hashes_from_disk()

    def _load_file_hashes_from_events(self):
        """Load file hashes and timestamps from cached events."""
        try:
            cached_messages = self.events_connection.get_all()
        except Exception:
            cached_messages = []

        if not cached_messages:
            return

        sorted_messages = sorted(cached_messages, key=lambda m: m.timestamp)

        for events_message in sorted_messages:
            for event in events_message.events:
                # Update last_event_timestamp_per_peer
                peer_email = event.datasite_email
                current_ts = self.last_event_timestamp_per_peer.get(peer_email)
                if current_ts is None or events_message.timestamp > current_ts:
                    self.last_event_timestamp_per_peer[peer_email] = (
                        events_message.timestamp
                    )

                # Update file_hashes
                path_key = Path(event.path_in_syftbox)
                if event.is_deleted:
                    if path_key in self.file_hashes:
                        del self.file_hashes[path_key]
                else:
                    self.file_hashes[path_key] = event.new_hash

    def _load_dataset_hashes_from_disk(self):
        """Scan local dataset directories and compute hashes to populate dataset_collection_hashes."""
        for collection_path in self._get_local_dataset_folders():
            content_hash = self._compute_local_dataset_hash(collection_path)
            if content_hash:
                self.dataset_collection_hashes[collection_path] = content_hash

    def get_collection_owner_email(self, collection_path: Path) -> str:
        """Extract the owner email from a collection path."""
        return collection_path.relative_to(self.syftbox_folder).parts[0]

    def get_collection_path(self, owner_email: str, tag: str) -> Path | None:
        """Get the full path to a collection for a given owner and tag."""
        if self.syftbox_folder is None or self.collection_subpath is None:
            return None
        return self.syftbox_folder / owner_email / self.collection_subpath / tag

    def _get_local_dataset_folders(self):
        """Yield paths to all local dataset folders."""
        if self.syftbox_folder is None or not self.syftbox_folder.exists():
            return
        if self.collection_subpath is None:
            return

        for email_dir in self.syftbox_folder.iterdir():
            if not email_dir.is_dir() or "@" not in email_dir.name:
                continue
            datasets_dir = email_dir / self.collection_subpath
            if not datasets_dir.exists():
                continue
            for tag_dir in datasets_dir.iterdir():
                if tag_dir.is_dir():
                    yield tag_dir

    def _compute_local_dataset_hash(self, collection_path: Path) -> str | None:
        """Compute content hash from local dataset files on disk."""
        from syft_client.sync.file_utils import compute_directory_hash

        return compute_directory_hash(collection_path)

    def clear_cache(self):
        self.events_connection.clear_cache()
        self.file_connection.clear_cache()
        self.file_hashes = {}
        self.last_sync = None
        self.peers = []
        self.current_check_point = None
        self.last_event_timestamp_per_peer = {}
        self.dataset_collection_hashes = {}

    @property
    def last_event_timestamp(self) -> float | None:
        if len(self.events_connection) == 0:
            return None
        return self.events_connection.get_latest().timestamp

    def sync_down(self, peer_email: str):
        # Use per-peer timestamp to avoid filtering out events from other peers
        peer_timestamp = self.last_event_timestamp_per_peer.get(peer_email)

        new_event_messages = self.connection_router.watcher_get_events_messages(
            peer_email=peer_email,
            since_timestamp=peer_timestamp,
        )
        for event_message in sorted(new_event_messages, key=lambda x: x.timestamp):
            self.apply_event_message(event_message)
            self.last_event_timestamp_per_peer[peer_email] = event_message.timestamp

        self.last_sync = datetime.now()

    def sync_down_parallel(
        self,
        peer_email: str,
        executor: ThreadPoolExecutor,
        download_fn: Callable[[str], FileChangeEventsMessage],
    ) -> int:
        """Sync with parallel file downloads. Returns the number of events applied."""
        peer_timestamp = self.last_event_timestamp_per_peer.get(peer_email)

        # Get file metadata (no download yet)
        file_metadatas = self.connection_router.watcher_get_outbox_file_metadatas(
            peer_email=peer_email,
            since_timestamp=peer_timestamp,
        )

        if not file_metadatas:
            # No new messages to download
            self.last_sync = datetime.now()
            return 0

        # Download all files in parallel
        file_ids = [m["file_id"] for m in file_metadatas]
        downloaded_messages = list(executor.map(download_fn, file_ids))

        # Apply in timestamp order
        event_count = 0
        for event_message in sorted(downloaded_messages, key=lambda x: x.timestamp):
            self.apply_event_message(event_message)
            self.last_event_timestamp_per_peer[peer_email] = event_message.timestamp
            event_count += len(event_message.events)

        self.last_sync = datetime.now()
        return event_count

    def apply_event_message(self, event_message: FileChangeEventsMessage):
        self.events_connection.write_file(
            event_message.message_filepath.as_string(), event_message
        )

        for event in event_message.events:
            # Normalize path to Path object for consistency in file_hashes dict
            path_key = Path(event.path_in_syftbox)

            if event.is_deleted:
                # Handle deletion
                self.file_connection.delete_file(str(event.path_in_syftbox))
                if path_key in self.file_hashes:
                    del self.file_hashes[path_key]
            else:
                # Handle create/update
                self.file_connection.write_file(
                    str(event.path_in_syftbox), event.content
                )
                self.file_hashes[path_key] = event.new_hash

    def get_cached_events(self) -> List[FileChangeEvent]:
        messages = self.events_connection.get_all()
        return [event for message in messages for event in message.events]

    def sync_down_if_needed(self, peer_email: str):
        if self.last_sync is None:
            self.sync_down(peer_email)

        time_since_last_sync = datetime.now() - self.last_sync
        if time_since_last_sync > timedelta(seconds=SECONDS_BEFORE_SYNCING_DOWN):
            self.sync_down(peer_email)

    def current_hash_for_file(self, path: str) -> int | None:
        for peer in self.peers:
            self.sync_down_if_needed(peer)
        return self.file_hashes.get(path, None)

    def _cleanup_stale_dataset_collections(
        self, peer_email: str, remote_collections: list[dict]
    ):
        """Remove locally cached dataset collections that no longer exist remotely."""
        remote_tags = {c["tag"] for c in remote_collections}

        for local_collection_path in list(self.dataset_collection_hashes.keys()):
            owner_email = self.get_collection_owner_email(local_collection_path)
            if owner_email != peer_email:
                continue
            if local_collection_path.name in remote_tags:
                continue
            del self.dataset_collection_hashes[local_collection_path]
            if self.syftbox_folder is not None:
                try:
                    rel_path = local_collection_path.relative_to(self.syftbox_folder)
                    self.file_connection.delete_directory(str(rel_path))
                except ValueError:
                    pass

    def sync_down_datasets(self, peer_email: str):
        """
        Sync dataset collections from peer.
        Separate from message sync. Uses hash to skip unchanged collections.
        """
        # Get list of collections shared with us (now returns list of dicts)
        collections = self.connection_router.watcher_list_dataset_collections()

        # Filter by peer
        peer_collections = [c for c in collections if c["owner_email"] == peer_email]

        self._cleanup_stale_dataset_collections(peer_email, peer_collections)

        for collection in peer_collections:
            owner_email = collection["owner_email"]
            tag = collection["tag"]
            content_hash = collection["content_hash"]

            # Check if hash changed - skip download if unchanged
            collection_path = self.get_collection_path(owner_email, tag)
            if collection_path is None:
                continue
            cached_hash = self.dataset_collection_hashes.get(collection_path)
            if cached_hash == content_hash:
                continue

            # Download collection files
            files = self.connection_router.watcher_download_dataset_collection(
                tag, content_hash, owner_email
            )

            # Write files to local cache (path relative to syftbox_folder)
            for file_name, content in files.items():
                rel_path = f"{owner_email}/{self.collection_subpath}/{tag}/{file_name}"
                self.file_connection.write_file(rel_path, content)

            # Update hash cache
            self.dataset_collection_hashes[collection_path] = content_hash

    def sync_down_datasets_parallel(
        self,
        peer_email: str,
        executor: ThreadPoolExecutor,
        download_fn: Callable[[str], bytes],
    ):
        """
        Sync dataset collections from peer with parallel file downloads.
        Downloads all files from all collections in a single parallel batch.
        """
        collections = self.connection_router.watcher_list_dataset_collections()
        peer_collections = [c for c in collections if c["owner_email"] == peer_email]

        self._cleanup_stale_dataset_collections(peer_email, peer_collections)

        # Gather all files to download across all collections
        all_downloads = []  # List of (collection_info, file_metadata)
        collections_to_update = []

        for collection in peer_collections:
            owner_email = collection["owner_email"]
            tag = collection["tag"]
            content_hash = collection["content_hash"]

            # Check if hash changed - skip download if unchanged
            collection_path = self.get_collection_path(owner_email, tag)
            if collection_path is None:
                continue
            cached_hash = self.dataset_collection_hashes.get(collection_path)
            if cached_hash == content_hash:
                continue

            # Get file metadata (no download yet)
            file_metadatas = (
                self.connection_router.watcher_get_dataset_collection_file_metadatas(
                    tag, content_hash, owner_email
                )
            )

            if not file_metadatas:
                continue

            collections_to_update.append(collection)
            for metadata in file_metadatas:
                all_downloads.append((collection, metadata))

        if not all_downloads:
            return

        # Download all files from all collections in parallel
        file_ids = [metadata["file_id"] for _, metadata in all_downloads]
        downloaded_contents = list(executor.map(download_fn, file_ids))

        # Write files to local cache (path relative to syftbox_folder)
        for (collection, metadata), content in zip(all_downloads, downloaded_contents):
            owner_email = collection["owner_email"]
            tag = collection["tag"]
            file_name = metadata["file_name"]
            rel_path = f"{owner_email}/{self.collection_subpath}/{tag}/{file_name}"
            self.file_connection.write_file(rel_path, content)

        # Update hash cache for all collections
        for collection in collections_to_update:
            collection_path = self.get_collection_path(
                collection["owner_email"], collection["tag"]
            )
            if collection_path is not None:
                self.dataset_collection_hashes[collection_path] = collection[
                    "content_hash"
                ]
