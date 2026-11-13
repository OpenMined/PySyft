import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Set

from pydantic import BaseModel, Field

from syft.sync.checkpoints.checkpoint import (
    DEFAULT_CHECKPOINT_EVENT_THRESHOLD,
    Checkpoint,
    CheckpointFile,
)
from syft.sync.checkpoints.rolling_state import RollingState
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)
from syft.sync.sync.caches.cache_file_writer_connection import (
    CacheFileConnection,
    FSFileConnection,
    InMemoryCacheFileConnection,
)
from syft.sync.sync.collection_spec import CollectionSyncSpec

logger = logging.getLogger(__name__)


def _is_within_peer_datasite(event: FileChangeEvent, peer_email: str) -> bool:
    """Whether ``event`` targets a file under ``<syftbox>/<peer_email>/``.

    The email is compared without case, as elsewhere. The path must stay
    inside the mirror: the file connection only guards the syftbox root, so a
    ``..`` segment could still reach another datasite's folder.
    """
    if event.datasite_email.casefold() != peer_email.casefold():
        return False
    path = event.path_in_datasite
    return not path.is_absolute() and ".." not in path.parts


def _peer_key(email: str) -> str:
    """``email`` without case, as peers' emails are compared elsewhere."""
    return email.casefold()


def _sent_at(msg_id: str) -> float:
    """The send time in the name of an events message."""
    return FileChangeEventsMessageFileName.from_string(msg_id).timestamp


def _events_within_peer_datasite(
    events: List[FileChangeEvent], peer_email: str
) -> List[FileChangeEvent]:
    """The events of ``events`` that target ``peer_email``'s datasite; log the rest."""
    accepted = []
    for event in events:
        if _is_within_peer_datasite(event, peer_email):
            accepted.append(event)
        else:
            logger.warning(
                f"Dropping event from {peer_email} aimed at "
                f"{event.path_in_syftbox}: a peer may only change files "
                "under its own datasite"
            )
    return accepted


SECONDS_BEFORE_SYNCING_DOWN = 0


class DataSiteWatcherCacheConfig(BaseModel):
    email: str = ""
    use_in_memory_cache: bool = True
    # Keep the state received from each DO in the DS's storage, and restore it
    # at startup. Off for a DS that can never restore it, as an enclave with
    # a new key at each boot.
    persist_watcher_state: bool = True
    syftbox_folder: Path | None = None
    # Collections to sync down (prefix + local subpath per collection type)
    collection_specs: list[CollectionSyncSpec] = []
    # Refuse a change or a delete of a file that the DS already has in an
    # immutable collection, so a DO cannot replace data it shared, such as a
    # private dataset in an enclave.
    freeze_private_datasets: bool = False


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
    # The msg_id of the newest events message received from each peer, keyed
    # by ``_peer_key``.
    last_msg_id_per_peer: Dict[str, str] = {}
    # The msg_ids of the events messages received from each peer, keyed by
    # ``_peer_key``. The connection lists back a bit before the last one for
    # late messages, and skips these.
    received_msg_ids_per_peer: Dict[str, Set[str]] = {}
    # Whether the state received from each peer is saved in the DS's storage.
    persist_watcher_state: bool
    # The changes from each peer since its checkpoint, keyed by ``_peer_key``.
    # The received ids stay in ``received_msg_ids_per_peer``; a saved copy
    # carries them.
    rolling_state_per_peer: Dict[str, RollingState] = {}
    # Base syftbox folder
    syftbox_folder: Path | None = None
    # Collections to sync down (prefix + local subpath per collection type)
    collection_specs: list[CollectionSyncSpec] = []
    # Cache of collection hashes: path -> content_hash
    collection_hashes: Dict[Path, str] = {}
    # Whether a file in an immutable collection keeps the first copy received.
    freeze_private_datasets: bool

    @classmethod
    def from_config(
        cls, config: DataSiteWatcherCacheConfig, connection_router: ConnectionRouter
    ):
        """``config`` holds plain values only. ``connection_router`` is live and
        shared with the other parts of the manager, so the caller builds it
        once and passes it in next to the config (see
        ``SyftboxManager.from_config``)."""
        if config.use_in_memory_cache:
            cache = cls(
                events_connection=InMemoryCacheFileConnection[FileChangeEvent](),
                file_connection=InMemoryCacheFileConnection[str](),
                connection_router=connection_router,
                syftbox_folder=config.syftbox_folder,
                collection_specs=config.collection_specs,
                persist_watcher_state=config.persist_watcher_state,
                freeze_private_datasets=config.freeze_private_datasets,
            )
        else:
            if config.syftbox_folder is None:
                raise ValueError("syftbox_folder is required for non-in-memory cache")

            cache = cls(
                file_connection=FSFileConnection(base_dir=config.syftbox_folder),
                connection_router=connection_router,
                syftbox_folder=config.syftbox_folder,
                collection_specs=config.collection_specs,
                persist_watcher_state=config.persist_watcher_state,
                freeze_private_datasets=config.freeze_private_datasets,
            )
            cache._load_collection_hashes_from_disk()
        if cache.persist_watcher_state:
            cache._restore_watcher_state()
        return cache

    def _load_collection_hashes_from_disk(self):
        """Scan local collection directories and compute hashes to populate collection_hashes."""
        for collection_path in self._get_local_collection_folders():
            content_hash = self._compute_local_collection_hash(collection_path)
            if content_hash:
                self.collection_hashes[collection_path] = content_hash

    def get_collection_owner_email(self, collection_path: Path) -> str:
        """Extract the owner email from a collection path."""
        return collection_path.relative_to(self.syftbox_folder).parts[0]

    def get_collection_path(
        self, owner_email: str, tag: str, local_subpath: Path
    ) -> Path | None:
        """Get the full path to a collection for a given owner, tag and subpath."""
        if self.syftbox_folder is None:
            return None
        return self.syftbox_folder / owner_email / local_subpath / tag

    def _get_local_collection_folders(self):
        """Yield paths to all local collection folders, in every layout of every spec."""
        if self.syftbox_folder is None or not self.syftbox_folder.exists():
            return

        for spec in self.collection_specs:
            if spec.owner_only:
                # Owner-only collections (e.g. private data) are never pulled from
                # peers, so the peer-facing watcher does not track them locally.
                continue
            for email_dir in self.syftbox_folder.iterdir():
                if not email_dir.is_dir() or "@" not in email_dir.name:
                    continue
                for layout in spec.layouts:
                    collections_dir = email_dir / layout.local_subpath
                    if not collections_dir.exists():
                        continue
                    for tag_dir in collections_dir.iterdir():
                        if tag_dir.is_dir():
                            yield tag_dir

    def _compute_local_collection_hash(self, collection_path: Path) -> str | None:
        """Compute content hash from local collection files on disk."""
        from syft.sync.file_utils import compute_directory_hash

        return compute_directory_hash(collection_path)

    def clear_cache(self):
        self.events_connection.clear_cache()
        self.file_connection.clear_cache()
        self.file_hashes = {}
        self.last_sync = None
        self.peers = []
        self.current_check_point = None
        self.last_msg_id_per_peer = {}
        self.received_msg_ids_per_peer = {}
        self.rolling_state_per_peer = {}
        self.collection_hashes = {}

    @property
    def last_event_timestamp(self) -> float | None:
        if len(self.events_connection) == 0:
            return None
        return self.events_connection.get_latest().timestamp

    def sync_down(self, peer_email: str) -> int:
        """Apply the events messages of ``peer_email`` not received yet: those
        after the last one, and any that landed late. Returns the number of
        events applied. A message that can never be read is passed over, so it
        does not hold back the later ones.

        A cache that saves its state saves the state of ``peer_email`` before
        it marks the messages processed."""
        key = _peer_key(peer_email)
        received = self.connection_router.watcher_receive_events_messages(
            peer_email,
            after=self.last_msg_id_per_peer.get(key),
            already_received=self.received_msg_ids_per_peer.get(key, frozenset()),
        )
        event_count = 0
        for raw, event_message in received:
            if event_message is not None:
                self.apply_event_message(event_message, peer_email)
                event_count += len(event_message.events)
            self._record_received(peer_email, raw.msg_id)
        if received and self.persist_watcher_state:
            self._save_watcher_state(peer_email)
        self.connection_router.mark_all_processed([raw for raw, _ in received])

        self.last_sync = datetime.now()
        return event_count

    def _record_received(self, peer_email: str, msg_id: str) -> None:
        """Remember ``msg_id``, and make it ``peer_email``'s last message when
        it is newer. A message that landed late is older than the last one."""
        key = _peer_key(peer_email)
        self.received_msg_ids_per_peer.setdefault(key, set()).add(msg_id)
        last = self.last_msg_id_per_peer.get(key)
        if last is None or _sent_at(msg_id) > _sent_at(last):
            self.last_msg_id_per_peer[key] = msg_id

    def apply_event_message(
        self, event_message: FileChangeEventsMessage, peer_email: str
    ) -> None:
        """Apply the events ``peer_email`` published in its outbox.

        Every event names its own target (``datasite_email`` and
        ``path_in_datasite``), and both fields come from the peer. A peer may
        only change files under its own datasite, mirrored locally at
        ``<syftbox>/<peer_email>/``, so an event aimed anywhere else is dropped
        before it reaches the local file connection.

        The record of the message and the saved state of ``peer_email`` hold
        only the accepted events, so a restore never writes a dropped event.
        """
        accepted = _events_within_peer_datasite(event_message.events, peer_email)
        saved = event_message.model_copy(update={"events": accepted})
        self.events_connection.write_file(saved.message_filepath.as_string(), saved)
        for event in accepted:
            if self._write_denied(event):
                continue
            self._apply_event(event)
            if self.persist_watcher_state:
                self._rolling_state_local(peer_email).add_event(event)

    def _write_denied(self, event: FileChangeEvent) -> bool:
        """Whether ``event`` changes or deletes a file that this cache already
        has in an immutable collection, with ``freeze_private_datasets`` on."""
        denied = (
            self.freeze_private_datasets
            and Path(event.path_in_syftbox) in self.file_hashes
            and any(
                spec.immutable
                and event.path_in_datasite.is_relative_to(spec.local_subpath)
                for spec in self.collection_specs
            )
        )
        if denied:
            logger.warning(
                f"Refused a change to {event.path_in_syftbox}: the file is in an "
                "immutable collection, and the DS has it already"
            )
        return denied

    def _apply_event(self, event: FileChangeEvent) -> None:
        path = str(event.path_in_syftbox)
        # Normalize path to Path object for consistency in file_hashes dict
        path_key = Path(event.path_in_syftbox)
        if event.is_deleted:
            self.file_connection.delete_file(path)
            self.file_hashes.pop(path_key, None)
        else:
            self.file_connection.write_file(path, event.content)
            self.file_hashes[path_key] = event.new_hash

    # --- State saved in the DS's storage ---

    def _restore_watcher_state(self) -> None:
        """Restore the files, their hashes and the received message ids of each
        peer with saved state. A Colab runtime reset loses the local copy.

        The state of a peer is saved under its ``_peer_key``, so two spellings
        of one email share it. The checkpoint goes first, because the rolling
        state holds the changes after it."""
        for peer_key in self.connection_router.watcher_peers_with_saved_state():
            self._restore_checkpoint(peer_key)
            self._restore_rolling_state(peer_key)

    def _restore_checkpoint(self, peer_key: str) -> None:
        """Write the files of the saved checkpoint of ``peer_key``, if any."""
        checkpoint = self.connection_router.watcher_load_checkpoint(peer_key)
        if checkpoint is None:
            return
        for file in checkpoint.files:
            self._restore_file(checkpoint.email, file)

    def _restore_rolling_state(self, peer_key: str) -> None:
        """Apply the changes of the saved rolling state of ``peer_key``, if any,
        and record the message ids it received."""
        rolling_state = self.connection_router.watcher_load_rolling_state(peer_key)
        if rolling_state is None:
            return
        for event in rolling_state.events:
            self._apply_event(event)
        for msg_id in rolling_state.received_msg_ids:
            self._record_received(peer_key, msg_id)
        self.rolling_state_per_peer[peer_key] = rolling_state.model_copy(
            update={"received_msg_ids": []}
        )

    def _restore_file(self, datasite_email: str, file: CheckpointFile) -> None:
        path_in_syftbox = Path(datasite_email) / file.path
        self.file_connection.write_file(str(path_in_syftbox), file.content)
        self.file_hashes[path_in_syftbox] = file.hash

    def _save_watcher_state(self, peer_email: str) -> None:
        """Save the rolling state of ``peer_email``. At
        ``DEFAULT_CHECKPOINT_EVENT_THRESHOLD`` paths, fold it into the checkpoint."""
        self._save_rolling_state(peer_email)
        if (
            self._rolling_state_local(peer_email).event_count
            >= DEFAULT_CHECKPOINT_EVENT_THRESHOLD
        ):
            self._fold_into_checkpoint(peer_email)

    def _fold_into_checkpoint(self, peer_email: str) -> None:
        """Merge the changes since the checkpoint into it, then save an empty
        rolling state. A deleted path leaves the checkpoint."""
        key = _peer_key(peer_email)
        rolling_state = self._rolling_state_local(peer_email)
        checkpoint = self.connection_router.watcher_load_checkpoint(key)
        if checkpoint is None:
            # Live events write under the email as the DO's events spell it, so
            # a restore writes the checkpoint files there too.
            checkpoint = Checkpoint(email=rolling_state.events[-1].datasite_email)
        folded = checkpoint.with_events(rolling_state.events)
        # The checkpoint goes first. After a stop between the two saves, the old
        # rolling state only repeats changes that the checkpoint has.
        self.connection_router.watcher_save_checkpoint(key, folded)
        self.rolling_state_per_peer[key] = RollingState(
            email=peer_email, base_checkpoint_timestamp=folded.timestamp
        )
        self._save_rolling_state(peer_email)

    def _save_rolling_state(self, peer_email: str) -> None:
        key = _peer_key(peer_email)
        received_msg_ids = sorted(self.received_msg_ids_per_peer[key])
        saved = self._rolling_state_local(peer_email).model_copy(
            update={"received_msg_ids": received_msg_ids}
        )
        self.connection_router.watcher_save_rolling_state(key, saved)

    def _rolling_state_local(self, peer_email: str) -> RollingState:
        """The changes from ``peer_email`` since its checkpoint, as held in
        memory. ``_save_rolling_state`` writes them to the DS's storage."""
        key = _peer_key(peer_email)
        if key not in self.rolling_state_per_peer:
            # 0.0 means no checkpoint yet, as for the DO's own rolling state.
            self.rolling_state_per_peer[key] = RollingState(
                email=peer_email, base_checkpoint_timestamp=0.0
            )
        return self.rolling_state_per_peer[key]

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

    def _select_collections_to_sync(
        self, spec: CollectionSyncSpec, collections: list[dict]
    ) -> list[dict]:
        """Keep one collection per tag: the newest layout this client reads.

        An owner publishes a collection once for each layout its audience reads.
        This client takes the newest of those it reads, and warns about the rest.
        """
        best: dict[tuple[str, str], dict] = {}
        for collection in collections:
            variant = collection.get("variant", "")
            if spec.rank_of(variant) < 0:
                logger.warning(
                    "Skipping '%s' from %s: it uses the %r layout of %s, which "
                    "this client does not read.",
                    collection["tag"],
                    collection["owner_email"],
                    variant,
                    spec.prefix,
                )
                continue
            key = (collection["owner_email"], collection["tag"])
            current = best.get(key)
            if current is None or spec.rank_of(variant) > spec.rank_of(
                current.get("variant", "")
            ):
                best[key] = collection
        return list(best.values())

    def _collection_path_of(self, spec: CollectionSyncSpec, collection: dict):
        """The local path a listed collection syncs to, or None without a syftbox."""
        layout = spec.layout_for(collection.get("variant", ""))
        if layout is None:
            return None
        return self.get_collection_path(
            collection["owner_email"], collection["tag"], layout.local_subpath
        )

    def _cleanup_stale_collections(
        self,
        spec: CollectionSyncSpec,
        peer_email: str,
        selected_collections: list[dict],
        remote_collections: list[dict],
    ):
        """Remove local collections this client no longer syncs from a peer.

        Two cases get removed: the owner deleted the collection, and this client
        now reads a newer layout of it. The second case would otherwise leave the
        older copy on disk, where a scan finds the same collection twice.

        A collection the owner still publishes, but in no layout this client
        reads, is kept: the copy on disk is then the last one this client could
        read. ``_select_collections_to_sync`` already logged why it is not
        refreshed.
        """
        layout_parents = set()
        if self.syftbox_folder is not None:
            layout_parents = {
                self.syftbox_folder / peer_email / layout.local_subpath
                for layout in spec.layouts
            }
        selected_paths = {
            self._collection_path_of(spec, c) for c in selected_collections
        }
        published = {(c["owner_email"], c["tag"]) for c in remote_collections}
        readable = {(c["owner_email"], c["tag"]) for c in selected_collections}

        for local_collection_path in list(self.collection_hashes.keys()):
            owner_email = self.get_collection_owner_email(local_collection_path)
            if owner_email != peer_email:
                continue
            # Only consider collections that live under one of this spec's layouts.
            if layout_parents and local_collection_path.parent not in layout_parents:
                continue
            if local_collection_path in selected_paths:
                continue
            # The last path segment is the tag, in every layout.
            collection = (owner_email, local_collection_path.name)
            if collection in published and collection not in readable:
                continue
            del self.collection_hashes[local_collection_path]
            if self.syftbox_folder is not None:
                try:
                    rel_path = local_collection_path.relative_to(self.syftbox_folder)
                    self.file_connection.delete_directory(str(rel_path))
                except ValueError:
                    pass

    def sync_down_collections(self, peer_email: str):
        """
        Sync collections from peer across all configured specs.
        Separate from message sync. Uses hash to skip unchanged collections.
        """
        for spec in self.collection_specs:
            if spec.owner_only:
                # Never pull owner-only collections (e.g. private data) from a peer.
                continue
            # Every layout the peer published; the prefix matches all of them.
            collections = self.connection_router.watcher_list_collections(spec.prefix)

            # Filter by peer, then take one layout for each collection
            published = [c for c in collections if c["owner_email"] == peer_email]
            peer_collections = self._select_collections_to_sync(spec, published)

            self._cleanup_stale_collections(
                spec, peer_email, peer_collections, published
            )

            for collection in peer_collections:
                owner_email = collection["owner_email"]
                tag = collection["tag"]
                content_hash = collection["content_hash"]
                layout = spec.layout_for(collection.get("variant", ""))

                # Check if hash changed - skip download if unchanged
                collection_path = self.get_collection_path(
                    owner_email, tag, layout.local_subpath
                )
                if collection_path is None:
                    continue
                cached_hash = self.collection_hashes.get(collection_path)
                if cached_hash == content_hash:
                    continue

                # Download collection files
                files = self.connection_router.watcher_download_collection(
                    spec.wire_prefix(layout.variant), tag, content_hash, owner_email
                )

                # Write files to local cache (path relative to syftbox_folder)
                for file_name, content in files.items():
                    rel_path = f"{owner_email}/{layout.local_subpath}/{tag}/{file_name}"
                    self.file_connection.write_file(rel_path, content)

                # Update hash cache
                self.collection_hashes[collection_path] = content_hash

    def sync_down_collections_parallel(
        self,
        peer_email: str,
        executor: ThreadPoolExecutor,
        download_fn: Callable[[str], bytes],
    ):
        """
        Sync collections from peer with parallel file downloads, across all specs.
        For each spec, downloads all files from all collections in a single
        parallel batch.
        """
        for spec in self.collection_specs:
            if spec.owner_only:
                # Never pull owner-only collections (e.g. private data) from a peer.
                continue
            collections = self.connection_router.watcher_list_collections(spec.prefix)
            published = [c for c in collections if c["owner_email"] == peer_email]
            peer_collections = self._select_collections_to_sync(spec, published)

            self._cleanup_stale_collections(
                spec, peer_email, peer_collections, published
            )

            # Gather all files to download across all collections for this spec
            all_downloads = []  # List of (collection_info, file_metadata)
            collections_to_update = []

            for collection in peer_collections:
                owner_email = collection["owner_email"]
                tag = collection["tag"]
                content_hash = collection["content_hash"]
                layout = spec.layout_for(collection.get("variant", ""))

                # Check if hash changed - skip download if unchanged
                collection_path = self.get_collection_path(
                    owner_email, tag, layout.local_subpath
                )
                if collection_path is None:
                    continue
                cached_hash = self.collection_hashes.get(collection_path)
                if cached_hash == content_hash:
                    continue

                # Get file metadata (no download yet)
                file_metadatas = (
                    self.connection_router.watcher_get_collection_file_metadatas(
                        spec.wire_prefix(layout.variant),
                        tag,
                        content_hash,
                        owner_email,
                    )
                )

                if not file_metadatas:
                    continue

                collections_to_update.append(collection)
                for metadata in file_metadatas:
                    all_downloads.append((collection, metadata))

            if not all_downloads:
                continue

            # Download all files from all collections in parallel
            file_ids = [metadata["file_id"] for _, metadata in all_downloads]
            downloaded_contents = list(executor.map(download_fn, file_ids))

            # Write files to local cache (path relative to syftbox_folder)
            for (collection, metadata), content in zip(
                all_downloads, downloaded_contents
            ):
                owner_email = collection["owner_email"]
                tag = collection["tag"]
                file_name = metadata["file_name"]
                subpath = spec.layout_for(collection.get("variant", "")).local_subpath
                rel_path = f"{owner_email}/{subpath}/{tag}/{file_name}"
                self.file_connection.write_file(rel_path, content)

            # Update hash cache for all collections in this spec
            for collection in collections_to_update:
                collection_path = self._collection_path_of(spec, collection)
                if collection_path is not None:
                    self.collection_hashes[collection_path] = collection["content_hash"]
