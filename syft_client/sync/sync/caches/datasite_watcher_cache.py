import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List

from pydantic import BaseModel, Field

from syft_client.sync.connections.base_connection import ConnectionConfig
from syft_client.sync.connections.connection_router import ConnectionRouter
from syft_client.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
)
from syft_client.sync.sync.caches.cache_file_writer_connection import (
    CacheFileConnection,
    FSFileConnection,
    InMemoryCacheFileConnection,
)

logger = logging.getLogger(__name__)

SECONDS_BEFORE_SYNCING_DOWN = 0


def _readable_dataset_protocol_versions() -> set[str]:
    """The dataset protocol versions that this client has a layout for."""
    from syft_datasets.protocolcodecs import CODECS

    return {
        protocol_version
        for codec_cls in CODECS
        for protocol_version in codec_cls.dataset_config_cls.protocol_versions
    }


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
    # Optional pre-write filter: (path_in_syftbox, is_delete) -> allow?
    # Return True to allow the write, False to deny it.
    pre_write_filter: Callable[[str, bool], bool] | None = None

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

    def _collection_rel_dir(
        self, owner_email: str, tag: str, protocol_version: str = "0"
    ) -> Path:
        """The local directory of a collection, relative to the SyftBox folder.

        Protocol 0 is flat. A later protocol adds its v<n> segment, so the files
        land where the metadata of that copy points.
        """
        from syft_datasets.config import protocol_dir_name

        base = Path(owner_email) / self.collection_subpath
        segment = protocol_dir_name(protocol_version)
        return base / segment / tag if segment else base / tag

    def get_collection_path(
        self, owner_email: str, tag: str, protocol_version: str = "0"
    ) -> Path | None:
        """Get the full path to a collection for a given owner, tag and protocol."""
        if self.syftbox_folder is None or self.collection_subpath is None:
            return None
        return self.syftbox_folder / self._collection_rel_dir(
            owner_email, tag, protocol_version
        )

    def _get_local_dataset_folders(self):
        """Yield paths to all local dataset folders, in every protocol layout."""
        from syft_datasets.config import is_protocol_dir_name

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
            for entry in datasets_dir.iterdir():
                if not entry.is_dir():
                    continue
                # A v<n> directory holds the tags of one protocol version.
                if is_protocol_dir_name(entry.name):
                    yield from (tag for tag in entry.iterdir() if tag.is_dir())
                else:
                    yield entry

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
                if self.pre_write_filter and not self.pre_write_filter(
                    str(event.path_in_syftbox), True
                ):
                    continue
                # Handle deletion
                self.file_connection.delete_file(str(event.path_in_syftbox))
                if path_key in self.file_hashes:
                    del self.file_hashes[path_key]
            else:
                if self.pre_write_filter and not self.pre_write_filter(
                    str(event.path_in_syftbox), False
                ):
                    continue
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

    def _select_collections_to_sync(self, collections: list[dict]) -> list[dict]:
        """Keep one collection for each dataset: the newest layout we can read.

        An owner publishes a dataset once for each protocol version that its
        audience reads. This client takes the newest of those that it reads, and
        ignores the rest.
        """
        readable = _readable_dataset_protocol_versions()
        best: dict[tuple[str, str], dict] = {}
        for collection in collections:
            protocol_version = collection.get("protocol_version", "0")
            if protocol_version not in readable:
                logger.warning(
                    "Skipping dataset '%s' from %s: it uses dataset protocol %s, "
                    "which this client does not read.",
                    collection["tag"],
                    collection["owner_email"],
                    protocol_version,
                )
                continue
            key = (collection["owner_email"], collection["tag"])
            current = best.get(key)
            if current is None or int(protocol_version) > int(
                current.get("protocol_version", "0")
            ):
                best[key] = collection
        return list(best.values())

    def _cleanup_stale_dataset_collections(
        self,
        peer_email: str,
        selected_collections: list[dict],
        remote_collections: list[dict],
    ):
        """Remove local collections that this client no longer syncs from a peer.

        Two cases get removed: the owner deleted the dataset, and this client now
        reads a newer layout of it. The second case would otherwise leave the
        older copy on disk, where a dataset scan finds the same dataset twice.

        A dataset that the owner still publishes, but in no layout this client
        reads, is kept. The copy on disk is then the last one this client could
        read, and a delete would take it away over an upgrade by someone else.
        ``_select_collections_to_sync`` already logged why it is not refreshed.
        """
        selected_paths = {
            self.get_collection_path(
                c["owner_email"], c["tag"], c.get("protocol_version", "0")
            )
            for c in selected_collections
        }
        published = {(c["owner_email"], c["tag"]) for c in remote_collections}
        readable = {(c["owner_email"], c["tag"]) for c in selected_collections}

        for local_collection_path in list(self.dataset_collection_hashes.keys()):
            owner_email = self.get_collection_owner_email(local_collection_path)
            if owner_email != peer_email:
                continue
            if local_collection_path in selected_paths:
                continue
            # The last path segment is the tag, in a flat and a v<n> layout both.
            dataset = (owner_email, local_collection_path.name)
            if dataset in published and dataset not in readable:
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

        # Filter by peer, then take one layout for each dataset
        published = [c for c in collections if c["owner_email"] == peer_email]
        peer_collections = self._select_collections_to_sync(published)

        self._cleanup_stale_dataset_collections(peer_email, peer_collections, published)

        for collection in peer_collections:
            owner_email = collection["owner_email"]
            tag = collection["tag"]
            content_hash = collection["content_hash"]
            protocol_version = collection.get("protocol_version", "0")

            # Check if hash changed - skip download if unchanged
            collection_path = self.get_collection_path(
                owner_email, tag, protocol_version
            )
            if collection_path is None:
                continue
            cached_hash = self.dataset_collection_hashes.get(collection_path)
            if cached_hash == content_hash:
                continue

            # Download collection files
            files = self.connection_router.watcher_download_dataset_collection(
                tag, content_hash, owner_email, protocol_version
            )

            # Write files to local cache (path relative to syftbox_folder)
            rel_dir = self._collection_rel_dir(owner_email, tag, protocol_version)
            for file_name, content in files.items():
                self.file_connection.write_file(str(rel_dir / file_name), content)

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
        published = [c for c in collections if c["owner_email"] == peer_email]
        peer_collections = self._select_collections_to_sync(published)

        self._cleanup_stale_dataset_collections(peer_email, peer_collections, published)

        # Gather all files to download across all collections
        all_downloads = []  # List of (collection_info, file_metadata)
        collections_to_update = []

        for collection in peer_collections:
            owner_email = collection["owner_email"]
            tag = collection["tag"]
            content_hash = collection["content_hash"]
            protocol_version = collection.get("protocol_version", "0")

            # Check if hash changed - skip download if unchanged
            collection_path = self.get_collection_path(
                owner_email, tag, protocol_version
            )
            if collection_path is None:
                continue
            cached_hash = self.dataset_collection_hashes.get(collection_path)
            if cached_hash == content_hash:
                continue

            # Get file metadata (no download yet)
            file_metadatas = (
                self.connection_router.watcher_get_dataset_collection_file_metadatas(
                    tag, content_hash, owner_email, protocol_version
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
            rel_dir = self._collection_rel_dir(
                collection["owner_email"],
                collection["tag"],
                collection.get("protocol_version", "0"),
            )
            self.file_connection.write_file(
                str(rel_dir / metadata["file_name"]), content
            )

        # Update hash cache for all collections
        for collection in collections_to_update:
            collection_path = self.get_collection_path(
                collection["owner_email"],
                collection["tag"],
                collection.get("protocol_version", "0"),
            )
            if collection_path is not None:
                self.dataset_collection_hashes[collection_path] = collection[
                    "content_hash"
                ]
