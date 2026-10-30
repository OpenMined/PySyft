import logging
from collections import deque
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from queue import Queue
from typing import List, Mapping, NamedTuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from syft_perms import SyftPermContext

from syft.sync.callback_mixin import BaseModelCallbackMixin
from syft.sync.checkpoints.checkpoint import (
    DEFAULT_COMPACTING_THRESHOLD,
    Checkpoint,
    CheckpointFile,
    IncrementalCheckpoint,
    compact_incremental_checkpoints,
)
from syft.sync.checkpoints.rolling_state import (
    RollingState,
    raise_for_later_version,
)
from syft.sync.connections.base_connection import FileCollection
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)
from syft.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.connections.types import RawMessage
from syft.sync.sync.caches.datasite_owner_cache import (
    DataSiteOwnerEventCache,
    DataSiteOwnerEventCacheConfig,
)
from syft.sync.sync.collection_spec import CollectionSyncSpec
from syft.sync.sync.constants import CACHE_DIR, ROLLING_STATE_FILENAME
from syft.sync.sync.delivery_tracker import (
    DeliveryPlan,
)
from syft.sync.utils.path_filters import is_normal_syncable_path
from syft.sync.utils.syftbox_utils import (
    create_event_timestamp,
    get_event_hash_from_content,
)

logger = logging.getLogger(__name__)


# Default threshold for creating incremental checkpoint from rolling state
DEFAULT_CHECKPOINT_EVENT_THRESHOLD = 50

# save_owner_state() uploads the rolling state once this many events were added
# since the last upload. At 1 the rolling state on Drive holds every change after
# each save_owner_state(), so the owner's event log is not written.
DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD = 1

# A delivery message holds about this much file content at most. A larger
# catch-up goes out in more than one message.
MAX_DELIVERY_MESSAGE_CONTENT = 50 * 1024 * 1024

# Minimum number of message files in a peer's outbox before
# compact_outbox_if_needed() will merge them into a single message.
MIN_MESSAGES_COMPACT = 20

# How often the owner tries an outbox message in one send pass before it gives
# up and logs an error. A file that a view tracks is planned again next sync.
MAX_SEND_ATTEMPTS = 3


class OwnerStateOnBackend(NamedTuple):
    """The owner's persisted state on the backend, in the order a restore
    applies it.

    ``log_messages`` are the event-log messages newer than the checkpoints and
    the rolling state. When there is no checkpoint and no rolling state, they
    are the whole log, and ``log_is_whole`` is True. A full checkpoint that
    carries no event cursor gets no log messages: it is the whole state.
    """

    full_checkpoint: Checkpoint | None
    incremental_checkpoints: List[IncrementalCheckpoint]
    # None when the backend has no rolling state, or an empty one.
    rolling_state: RollingState | None
    log_messages: List[FileChangeEventsMessage]
    log_is_whole: bool

    def events_messages(self) -> List[FileChangeEventsMessage]:
        """Every event since the full checkpoint, in order."""
        messages = [
            FileChangeEventsMessage(events=checkpoint.events)
            for checkpoint in self.incremental_checkpoints
        ]
        if self.rolling_state is not None:
            messages.append(FileChangeEventsMessage(events=self.rolling_state.events))
        return messages + self.log_messages


def _last_persisted_event_timestamp(
    full_checkpoint: Checkpoint | None,
    incrementals: List[IncrementalCheckpoint],
    rolling_state: RollingState | None = None,
) -> float | None:
    """The timestamp of the last event that the checkpoints and the rolling
    state hold. The event log is read after it."""
    since = full_checkpoint.last_event_timestamp if full_checkpoint else None
    for checkpoint in incrementals:
        for event in checkpoint.events:
            since = _later_timestamp(since, event.timestamp)
    if rolling_state is not None:
        since = _later_timestamp(since, rolling_state.last_event_timestamp)
    return since


def _incrementals_after(
    full_checkpoint: Checkpoint | None, incrementals: List[IncrementalCheckpoint]
) -> List[IncrementalCheckpoint]:
    """The incrementals that ``full_checkpoint`` does not hold yet.

    A full checkpoint is built after the incrementals it holds, so an older
    incremental is already in it. A failed delete can leave one on the backend.
    """
    if full_checkpoint is None:
        return incrementals
    return [cp for cp in incrementals if cp.timestamp > full_checkpoint.timestamp]


def _later_timestamp(current: float | None, candidate: float | None) -> float | None:
    """The later of two event timestamps, either of which may be missing."""
    if candidate is None:
        return current
    if current is None or candidate > current:
        return candidate
    return current


def same_email(left: str | None, right: str | None) -> bool:
    """True if both name the same address. A mail address has no case."""
    if left is None or right is None:
        return False
    return left.casefold() == right.casefold()


class DatasiteOwnerSyncerConfig(BaseModel):
    email: str
    syftbox_folder: Path
    write_files: bool = True
    # Whether to keep the owner-only state that exists solely to restore this
    # datasite later: the append-only event log, the rolling state and the
    # checkpoints. Off for a datasite that can never restore - an enclave gets
    # ephemeral keys and wipes its state each boot - so those writes are pure
    # cost. Peer-facing state (outbox, collections, peers) is unaffected.
    persist_owner_state: bool = True
    # Full path to collections folder - must be provided explicitly
    collections_folder: Path | None = None
    # Collection sync specs (public prefix + local subpath). Empty for a bare
    # sync engine; the domain layer (e.g. syft-rds) supplies the concrete specs.
    collection_specs: List[CollectionSyncSpec] = []
    cache_config: DataSiteOwnerEventCacheConfig = Field(
        default_factory=DataSiteOwnerEventCacheConfig
    )


class OutgoingMessage(NamedTuple):
    recipient: str
    message: FileChangeEventsMessage
    # Failed sends of this message so far.
    attempts: int = 0


def _batches_by_content_size(
    events: list[FileChangeEvent], limit: int
) -> list[list[FileChangeEvent]]:
    """Split ``events`` in order into lists of at most ``limit`` content bytes.

    An event larger than ``limit`` gets a list of its own.
    """
    batches: list[list[FileChangeEvent]] = []
    size = 0
    for event in events:
        event_size = len(event.content or b"")
        if not batches or size + event_size > limit:
            batches.append([])
            size = 0
        batches[-1].append(event)
        size += event_size
    return batches


class DatasiteOwnerSyncer(BaseModelCallbackMixin):
    """Responsible for downloading files and checking permissions"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    event_cache: DataSiteOwnerEventCache = Field(
        default_factory=lambda: DataSiteOwnerEventCache()
    )
    write_files: bool = True
    persist_owner_state: bool = True
    connection_router: ConnectionRouter
    initial_sync_done: bool = False
    email: str
    syftbox_folder: Path
    perm_context: SyftPermContext
    # Full path to collections folder
    collections_folder: Path | None = None
    # Collection sync specs (public prefix + local subpath), supplied by rds.
    collection_specs: List[CollectionSyncSpec] = []

    syftbox_events_queue: Queue[FileChangeEventsMessage] = Field(default_factory=Queue)
    outbox_queue: Queue[OutgoingMessage] = Field(default_factory=Queue)

    _executor: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=10)
    )
    # Cache of collections shared with "any" - list of (tag, content_hash) tuples
    _any_shared_collections: List[tuple] = PrivateAttr(default_factory=list)

    # In-memory rolling state for tracking events since last checkpoint
    _rolling_state: RollingState | None = PrivateAttr(default=None)
    # Counter for events since last rolling state upload
    _events_since_rolling_state_upload: int = PrivateAttr(default=0)

    @classmethod
    def from_config(
        cls, config: DatasiteOwnerSyncerConfig, connection_router: ConnectionRouter
    ):
        """``config`` holds plain values only. ``connection_router`` is live and
        shared with the other parts of the manager, so the caller builds it
        once and passes it in next to the config (see
        ``SyftboxManager.from_config``)."""
        # Ensure cache config has the same collections_folder (both are now full paths)
        if config.collections_folder is not None:
            config.cache_config.collections_folder = config.collections_folder

        datasite = config.syftbox_folder / config.email
        datasite.mkdir(parents=True, exist_ok=True)

        return cls(
            event_cache=DataSiteOwnerEventCache.from_config(config.cache_config),
            write_files=config.write_files,
            persist_owner_state=config.persist_owner_state,
            connection_router=connection_router,
            email=config.email,
            syftbox_folder=config.syftbox_folder,
            perm_context=SyftPermContext(datasite=datasite),
            collections_folder=config.collections_folder,
            collection_specs=config.collection_specs,
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
            state = RollingState.model_validate_json(path.read_text())
            raise_for_later_version(state.version)
        except (ValueError, OSError) as e:
            # A later client wrote this file, or it is damaged. The caller falls
            # back to a download of all events.
            print(f"Warning: could not load the local rolling state: {e}")
            return
        self._rolling_state = state

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
        """Record the owner's changes, pull the proposed changes of
        ``peer_emails``, then send each of them what it does not hold yet, in
        one pass.

        Accepted proposals and the owner's own changes go out the same way: to
        every peer that may read the file. A proposal that fails is logged as
        an error and does not hold back the send.
        """
        if self.persist_owner_state:
            self._load_rolling_state()
        try:
            if not self.initial_sync_done:
                self.pull_initial_state()
            # The one permission load of this pass. Each step below reads it.
            self.perm_context._reload()
            # Before the local changes are recorded, so that a file changed
            # while the owner was offline is not seeded as held.
            self._seed_views_if_missing(peer_emails)
            # Before the proposals, so that a proposal based on an older
            # version of a file the owner changed is refused as outdated.
            if recompute_hashes:
                self._record_local_changes()
            for peer_email in peer_emails:
                self._pull_proposals(peer_email)
            # The owner's log goes out before any peer gets a change.
            self.save_owner_state()
            self.deliver_file_changes(peer_emails)
        finally:
            if self.persist_owner_state:
                self._save_rolling_state()

    def _pull_proposals(self, peer_email: str) -> None:
        """Apply every proposed change waiting from ``peer_email``, then archive
        them all in one go.

        A proposal that cannot be read or applied is archived with a warning,
        so it never holds back the ones after it. Proposals stay in the inbox
        only while the peer is not set up yet or Drive cannot be reached; that
        is logged, so it does not hold back the other peers or the owner's
        changes.
        """
        proposed_count = 0
        try:
            received = self.connection_router.owner_receive_proposed_messages(
                peer_email
            )
            for raw, message in received:
                if self._apply_proposed_message(raw, message):
                    proposed_count += len(message.proposed_file_changes)
            self.connection_router.owner_mark_all_processed([r for r, _ in received])
        except SyftPeerNotReadyError as e:
            warnings.warn(f"Messages from {peer_email} stay in the inbox: {e}")
        except Exception as e:
            logger.error(f"Proposed changes from {peer_email} failed: {e!r}")
        if proposed_count:
            print(f"Received {proposed_count} proposed change(s) from {peer_email}")

    def download_events_message_by_id_with_connection(
        self, events_message_id: str
    ) -> FileChangeEventsMessage:
        """One event-log message, on its own connection (safe to run in parallel)."""
        return self.connection_router.download_events_message_by_id(events_message_id)

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

        Two halves:
        1. The owner's own state - checkpoints, rolling state, and the events
           newer than those. Skipped entirely when `persist_owner_state` is
           off: nothing was ever written, so there is nothing to read back.
        2. The owner's collections, which are shared with peers rather than
           owner-only, so they are restored either way.
        """
        if self.persist_owner_state:
            self._restore_owner_state()

        # Restore ALL registered collections (public + private) generically. Each
        # spec's flags decide the behaviour: `immutable` picks mirror vs restore-only;
        # the local destination comes from the spec's local_subpath.
        self._pull_collections_for_initial_sync()

        self.initial_sync_done = True

    def _restore_owner_state(self) -> None:
        """Rebuild the cache from the owner's persisted state on the backend."""
        state = self.fetch_owner_state(
            whole_log_since=self.event_cache.latest_cached_timestamp
        )
        self._apply_full_checkpoint(state.full_checkpoint)
        events = self._apply_incremental_checkpoints(state.incremental_checkpoints)
        events += self._apply_rolling_state(state)
        self._apply_log_messages(state)

        # The restore above only populates file_hashes and file_connection, not
        # events_messages_connection, which get_cached_events() reads from.
        if events and not self.event_cache.get_cached_events():
            self._write_events_to_messages_cache(events)

    def fetch_owner_state(
        self, whole_log_since: float | None = None
    ) -> OwnerStateOnBackend:
        """Download the owner's persisted state: the full checkpoint, the
        incrementals, the rolling state, and the event-log messages after them.

        ``whole_log_since`` limits the log when it is read whole, because
        there is no checkpoint and no rolling state.
        """
        router = self.connection_router
        full_checkpoint = router.get_latest_checkpoint()
        incrementals = _incrementals_after(
            full_checkpoint, router.get_all_incremental_checkpoints()
        )
        rolling_state = router.get_rolling_state()
        if rolling_state is not None and rolling_state.event_count == 0:
            rolling_state = None
        since = _last_persisted_event_timestamp(
            full_checkpoint, incrementals, rolling_state
        )
        log_is_whole = since is None and full_checkpoint is None and not incrementals
        if since is not None:
            log_messages = router.get_events_messages_since_timestamp(since)
        elif log_is_whole:
            log_messages = self.get_all_accepted_events_messages(whole_log_since)
        else:
            log_messages = []
        return OwnerStateOnBackend(
            full_checkpoint, incrementals, rolling_state, log_messages, log_is_whole
        )

    def _apply_full_checkpoint(self, checkpoint: Checkpoint | None) -> None:
        """Restore a full snapshot."""
        if checkpoint is None:
            return
        print(f"Found full checkpoint with {len(checkpoint.files)} files, restoring...")
        self.event_cache.apply_checkpoint(checkpoint, write_files=self.write_files)

    def _apply_incremental_checkpoints(
        self, checkpoints: List[IncrementalCheckpoint]
    ) -> List[FileChangeEvent]:
        """Apply incrementals in order. Returns their events."""
        if not checkpoints:
            return []
        print(f"Found {len(checkpoints)} incremental checkpoints, applying...")
        events: List[FileChangeEvent] = []
        for inc_cp in checkpoints:
            self._apply_incremental_checkpoint_to_cache(inc_cp)
            events.extend(inc_cp.events)
        return events

    def _apply_rolling_state(self, state: OwnerStateOnBackend) -> List[FileChangeEvent]:
        """Apply the rolling state, or start an empty one when there is none.
        Returns its events."""
        rolling_state = state.rolling_state
        if rolling_state is None:
            since = _last_persisted_event_timestamp(
                state.full_checkpoint, state.incremental_checkpoints
            )
            self._rolling_state = RollingState(
                email=self.email, base_checkpoint_timestamp=since or 0.0
            )
            return []
        print(
            f"Found rolling state with {rolling_state.event_count} events, applying..."
        )
        self._apply_rolling_state_to_cache(rolling_state)
        self._rolling_state = rolling_state
        return rolling_state.events

    def _apply_log_messages(self, state: OwnerStateOnBackend) -> None:
        """Cache the event-log messages after the checkpoints and the rolling
        state. Only those that follow a checkpoint or a rolling state go into
        the rolling state."""
        if state.log_is_whole:
            print("No checkpoints found, downloading all events...")
        elif state.log_messages:
            print(
                f"Downloading {len(state.log_messages)} events since "
                "checkpoint/rolling state..."
            )
        for events_message in state.log_messages:
            self.event_cache.add_events_message_to_local_cache(events_message)
            if not state.log_is_whole:
                self._add_events_to_rolling_state(events_message)

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

    def _collection_local_dir(
        self, spec: CollectionSyncSpec, tag: str, variant: str = ""
    ) -> Path:
        """Local directory one layout of a collection restores to."""
        layout = spec.layout_for(variant)
        subpath = layout.local_subpath if layout else spec.local_subpath
        return self.syftbox_folder / self.email / subpath / tag

    def _pull_collections_for_initial_sync(self):
        """Restore the owner's own collections from the sync backend on connect.

        Iterates every registered spec (public, private, or any future kind). For
        each it lists the owner's collections, refreshes the _any_shared_collections
        cache, filters by the spec's download policy, and restores to disk.
        """
        for spec in self.collection_specs:
            collections = (
                self.connection_router.owner_list_all_collections_with_permissions(
                    spec.prefix
                )
            )
            if not collections:
                continue

            self._update_any_shared_collections_cache(spec, collections)

            collections_to_download = self._filter_collections_needing_download(
                collections, spec
            )
            self._download_collections_parallel(collections_to_download, spec)

    @property
    def any_shared_collections(self) -> List[tuple]:
        """Collections shared with "any" as (wire_prefix, tag, content_hash) triples.

        One entry per layout: each layout is its own collection on the wire, so
        sharing one with a new peer needs the prefix that names it.

        Read-only view: mutate via ``register_any_shared_collection``.
        """
        return list(self._any_shared_collections)

    def register_any_shared_collection(
        self, wire_prefix: str, tag: str, content_hash: str
    ) -> None:
        """Record one layout of a collection as shared-with-"any" (deduplicated)."""
        entry = (wire_prefix, tag, content_hash)
        if entry not in self._any_shared_collections:
            self._any_shared_collections.append(entry)

    def _update_any_shared_collections_cache(
        self, spec: CollectionSyncSpec, collections: list[FileCollection]
    ):
        """Populate the any-shared cache from collections with 'any' permission."""
        for collection in collections:
            if collection.has_any_permission:
                self.register_any_shared_collection(
                    spec.wire_prefix(collection.variant),
                    collection.tag,
                    collection.content_hash,
                )

    def _filter_collections_needing_download(
        self, collections: list[FileCollection], spec: CollectionSyncSpec
    ) -> list[FileCollection]:
        """Return the collections that need downloading, per the spec's policy.

        * immutable (restore-only): download only when absent locally — the local
          copy is authoritative and must never be overwritten by the backup.
        * mutable (mirror): download when the remote content-hash differs from ours.
        """
        from syft.sync.file_utils import compute_directory_hash

        result = []
        for collection in collections:
            local_dir = self._collection_local_dir(
                spec, collection.tag, collection.variant
            )

            if spec.immutable:
                if not local_dir.exists() or not any(local_dir.iterdir()):
                    result.append(collection)
                continue

            # Mirror: use the cached hash first, else compute it from local files.
            cached_hash = self.event_cache.get_collection_hash(
                collection.tag, collection.variant
            )
            if cached_hash is None and local_dir.exists():
                cached_hash = compute_directory_hash(local_dir)
                if cached_hash is not None:
                    self.event_cache.set_collection_hash(
                        collection.tag, cached_hash, collection.variant
                    )

            if cached_hash != collection.content_hash:
                result.append(collection)
        return result

    def _download_collections_parallel(
        self, collections: list[FileCollection], spec: CollectionSyncSpec
    ):
        """Download all files from collections in parallel and write to disk."""
        if not collections:
            return

        # Fetch file metadatas for all collections in parallel. Each collection
        # names its own layout, so the prefix comes from the collection.
        all_file_metadatas = list(
            self._executor.map(
                partial(self._get_file_metadatas_with_new_connection, spec=spec),
                collections,
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

        # Write all files to disk under the layout's local subpath.
        for (collection, metadata), content in zip(all_downloads, downloaded_contents):
            local_dir = self._collection_local_dir(
                spec, collection.tag, collection.variant
            )
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / metadata["file_name"]).write_bytes(content)

        # Update cached hashes ONLY for mutable specs
        if not spec.immutable:
            for collection in collections:
                self.event_cache.set_collection_hash(
                    collection.tag, collection.content_hash, collection.variant
                )

        # Notify the domain layer (e.g. syft-rds) that these collections were restored,
        # so it can run any collection-specific post-processing
        for collection in collections:
            self._emit(
                "collection_restored",
                spec.prefix,
                collection.tag,
                self._collection_local_dir(spec, collection.tag, collection.variant),
            )

    def _get_file_metadatas_with_new_connection(
        self, collection: FileCollection, spec: CollectionSyncSpec
    ) -> list:
        """Get file metadatas for a collection using a new connection for thread safety."""
        collections = (
            self.connection_router.connections.collections_con.clone_for_thread()
        )
        return collections.list_files(
            spec.wire_prefix(collection.variant),
            tag=collection.tag,
            content_hash=collection.content_hash,
            owner_email=self.email,
        )

    def _download_file_with_new_connection(self, file_id: str) -> bytes:
        """Download a file using a new connection for thread safety."""
        collections = (
            self.connection_router.connections.collections_con.clone_for_thread()
        )
        return collections.fetch_file(file_id)

    def _create_current_filestate_event(self, path: str) -> FileChangeEvent:
        """An event that sets ``path`` on a peer to its current local state.

        A file missing from the cache gives a deletion. A read error raises.
        """
        try:
            content = self.event_cache.file_connection.read_file(path)
        except (KeyError, FileNotFoundError):
            # The in-memory cache raises KeyError, the filesystem cache
            # FileNotFoundError.
            return self._deletion_event(path)
        timestamp = create_event_timestamp()
        return FileChangeEvent(
            id=uuid4(),
            path_in_datasite=Path(path),
            content=content,
            new_hash=get_event_hash_from_content(content),
            submitted_timestamp=timestamp,
            timestamp=timestamp,
            datasite_email=self.email,
        )

    def _deletion_event(self, path: str) -> FileChangeEvent:
        """An event that removes ``path`` from a peer."""
        timestamp = create_event_timestamp()
        return FileChangeEvent(
            id=uuid4(),
            path_in_datasite=Path(path),
            content=None,
            new_hash=None,
            submitted_timestamp=timestamp,
            timestamp=timestamp,
            datasite_email=self.email,
            is_deleted=True,
        )

    def _record_local_changes(self) -> None:
        """Update the file hashes from the local files, and add the changes to
        the owner's persisted state. The caller then calls ``save_owner_state``."""
        file_change_events_message = self.event_cache.process_local_file_changes()
        if file_change_events_message is not None:
            self._queue_for_owner_state(file_change_events_message)

    def _is_syncable(self, path: str) -> bool:
        """Whether a peer can get ``path`` by a delivery plan."""
        collections_rel = self.event_cache.collections_relative_path()
        return is_normal_syncable_path(path, collections_path=collections_rel)

    def _syncable_hashes(self) -> dict[str, str]:
        """The current hash of every file that a peer can get, by path."""
        return {
            str(path): file_hash
            for path, file_hash in self.event_cache.file_hashes.items()
            if self._is_syncable(str(path))
        }

    def _readable_files_by_peer(
        self, peer_emails: list[str]
    ) -> dict[str, dict[str, str]]:
        """For each peer, the current hash of every syncable file it may read.

        Maps each peer email to ``{path: hash}``. The caller reloads the
        permissions first.
        """
        syncable = self._syncable_hashes()
        # peer email -> {path: hash}
        return {
            peer: {
                p: h for p, h in syncable.items() if self.perm_context.can_read(p, peer)
            }
            for peer in peer_emails
        }

    def _may_hold(self, peer: str, path: str, syncable: Mapping[str, str]) -> bool:
        """Whether a view of ``peer`` may hold ``path``: a file in ``syncable``
        that ``peer`` may read."""
        return path in syncable and self.perm_context.can_read(path, peer)

    def _seed_views_if_missing(self, peer_emails: list[str]) -> None:
        """Seed the views one time, for an owner that has file hashes but no views.

        Each approved peer, and each peer in ``peer_emails``, is taken to hold
        every file it may read now. The caller reloads the permissions first.
        """
        delivery_tracker = self.event_cache.delivery_tracker
        if delivery_tracker.is_seeded():
            return
        approved = [p.email for p in self.connection_router.peer_store.approved_peers]
        peers_to_seed = sorted({*approved, *peer_emails})
        delivery_tracker.seed(self._readable_files_by_peer(peers_to_seed))

    def deliver_file_changes(self, peer_emails: list[str]) -> None:
        """Send each peer the files its view lacks, and deletes for the files it
        may no longer have. The caller loads the permissions first."""
        delivery_tracker = self.event_cache.delivery_tracker
        plans = delivery_tracker.plan(self._readable_files_by_peer(peer_emails))
        # Build every message before the first one is queued, so a read error
        # leaves no message of this plan in the queue.
        messages = [
            OutgoingMessage(peer, FileChangeEventsMessage(events=batch))
            for peer, plan in plans.items()
            for batch in _batches_by_content_size(
                self._plan_events(plan), MAX_DELIVERY_MESSAGE_CONTENT
            )
        ]
        for message in messages:
            self.outbox_queue.put(message)
        self._send_outbox_messages()

    def _plan_events(self, plan: DeliveryPlan) -> list[FileChangeEvent]:
        """The events that carry out ``plan``: its sends, then its deletes."""
        events = [self._create_current_filestate_event(p) for p in plan.send]
        return events + [self._deletion_event(p) for p in plan.delete]

    def _apply_proposed_message(
        self, received: RawMessage, message: ProposedFileChangesMessage
    ) -> bool:
        """Apply one proposal; False, with a warning, when applying it failed.
        Either way the caller archives it, so it does not hold back the next."""
        # The sender is the peer whose inbox held the message. The p2p connection
        # read the message from that peer's folder and checked the signature
        # against that peer's key. The body is sender-supplied, so a claim
        # there that disagrees is a spoof attempt.
        sender_email = received.sender
        try:
            if same_email(message.sender_email, sender_email):
                self.handle_proposed_filechange_events_message(sender_email, message)
            else:
                logger.warning(
                    f"Dropped a proposed change from {sender_email}: the message "
                    f"claims sender {message.sender_email!r}."
                )
            return True
        except Exception as e:
            warnings.warn(
                f"Archived proposed change {received.msg_id} from {sender_email}: "
                f"applying it failed: {e!r}"
            )
            return False

    def check_write_permission(self, sender_email: str, path: str) -> bool:
        """Check if sender has write access to the given path."""
        self.perm_context._reload()
        return self.perm_context.open(path).has_write_access(sender_email)

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
            sender_email=sender_email,
            proposed_file_changes=allowed_changes,
        )

        accepted_events_message = self.event_cache.process_proposed_events_message(
            filtered_message
        )
        # The next plan sends the accepted change to every peer that may read
        # the file, the submitter included.
        if accepted_events_message is not None:
            self._queue_for_owner_state(accepted_events_message)

    def process_events_for_own_state_and_peers(
        self, recipients: list[str], file_change_events_message: FileChangeEventsMessage
    ) -> None:
        """Add ``file_change_events_message`` to the owner state, save that
        state, then send the message to each of ``recipients``."""
        self._queue_for_owner_state(file_change_events_message)
        self.save_owner_state()
        for recipient in recipients:
            self.outbox_queue.put(
                OutgoingMessage(recipient, file_change_events_message)
            )
        self._send_outbox_messages()

    def _queue_for_owner_state(self, events_message: FileChangeEventsMessage) -> None:
        """Add ``events_message`` to the owner's persisted state, read back only
        to restore this datasite.

        It goes into the rolling state, and into the owner's event log unless
        the rolling state holds every change after each ``save_owner_state``. A rolling state
        that is not loaded yet is loaded from disk.

        Raises:
            RuntimeError: when there is no rolling state on disk either, because
                this datasite never synced here.
        """
        if not self.persist_owner_state:
            return
        if self._rolling_state is None:
            self._load_rolling_state()
        if self._rolling_state is None:
            raise RuntimeError(
                "Sync this datasite before changing it: its owner state is not loaded."
            )
        if DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD > 1:
            self.syftbox_events_queue.put(events_message)
        self._add_events_to_rolling_state(events_message)

    def save_owner_state(self):
        """Write the owner's event log, and upload and save its rolling state."""
        # TODO: make this atomic
        while not self.syftbox_events_queue.empty():
            file_change_events_message = self.syftbox_events_queue.get()
            self.connection_router.owner_write_events_message_to_syftbox(
                file_change_events_message
            )
        if self.persist_owner_state:
            self._upload_rolling_state_if_due()
            # Also outside sync(), whose start reloads the rolling state from disk.
            self._save_rolling_state()

    def _send_outbox_messages(self) -> None:
        """Send the queued messages, and record each sent one in its peer's view.

        Nothing raises, and a failed send does not stop the others. A failed
        message goes to the back of this pass, and after ``MAX_SEND_ATTEMPTS``
        tries it is dropped with an error in the log. A peer that is not ready
        gets none of its messages in this pass, with a warning. A file that a
        view tracks is planned again at the next sync.
        """
        pending = deque()
        while not self.outbox_queue.empty():
            pending.append(self.outbox_queue.get())
        not_ready = set()
        while pending:
            entry = pending.popleft()
            if entry.recipient in not_ready:
                continue
            try:
                self.connection_router.owner_write_event_messages_to_outbox(
                    entry.recipient, entry.message
                )
            except SyftPeerNotReadyError as e:
                not_ready.add(entry.recipient)
                warnings.warn(f"Messages for {entry.recipient} are not sent: {e}")
            except Exception as e:
                pending.extend(self._retry_or_give_up(entry, e))
            else:
                self._record_sent(entry.recipient, entry.message)

    def _retry_or_give_up(
        self, entry: OutgoingMessage, error: Exception
    ) -> list[OutgoingMessage]:
        """The entry to try again for a failed send, if any try is left."""
        name = entry.message.message_filepath.as_string()
        attempts = entry.attempts + 1
        if attempts >= MAX_SEND_ATTEMPTS:
            logger.error(
                f"Gave up sending {name} to {entry.recipient} after {attempts} tries: {error!r}"
            )
            return []
        logger.info(
            f"Send of {name} to {entry.recipient} failed, trying again: {error!r}"
        )
        return [entry._replace(attempts=attempts)]

    def _record_sent(self, recipient: str, message: FileChangeEventsMessage) -> None:
        """Add the changes of a sent message to ``recipient``'s view.

        A delete always counts. A file counts only if it is syncable and
        ``recipient`` may read it, so a later permission change sends a delete
        for it. Other files, such as a private dataset sent to an enclave, are
        not in the view.
        """
        syncable = self._syncable_hashes()
        changes = {}
        for event in message.events:
            path = str(event.path_in_datasite)
            if event.is_deleted:
                changes[path] = None
            elif self._may_hold(recipient, path, syncable):
                changes[path] = event.new_hash
        self.event_cache.delivery_tracker.record_send(recipient, changes)

    def _download_outbox_message_with_new_connection(
        self, file_id: str
    ) -> FileChangeEventsMessage:
        """Download and decompress one outbox events-message file."""
        raw = self.connection_router.download_outbox_message(file_id)
        return FileChangeEventsMessage.from_compressed_data(raw)

    def compact_outbox_if_needed(
        self,
        recipient_email: str,
        min_messages: int = MIN_MESSAGES_COMPACT,
    ) -> int:
        """Merge accumulated event-message files in a recipient's outbox into fewer.

        Compaction downloads them, concatenates their events, writes them again
        as messages of at most ``MAX_DELIVERY_MESSAGE_CONTENT`` of file content,
        and deletes the originals.

        Returns the number of source messages compacted (0 if below threshold,
        if no outbox folder, or if peer encryption is on — DO can't read
        peer-encrypted bytes back to merge).
        """
        peer_store = self.connection_router.peer_store
        if peer_store.pair_encrypts(peer_store.get_cached_peer(recipient_email)):
            logger.debug(
                f"outbox compaction for {recipient_email} skipped — "
                f"peer encryption is not supported yet"
            )
            return 0

        metas = self.connection_router.owner_get_outbox_file_metadatas(recipient_email)

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

        # Write the merged messages FIRST. A DS that pulls between this and
        # the delete sees `merged ∪ some originals`; its file_hashes cache
        # makes duplicate events no-op writes. The size cap of a delivery
        # message holds here too.
        for batch in _batches_by_content_size(
            merged_events, MAX_DELIVERY_MESSAGE_CONTENT
        ):
            self.connection_router.owner_write_event_messages_to_outbox(
                recipient_email, FileChangeEventsMessage(events=batch)
            )
        self.connection_router.owner_delete_outbox_messages(file_ids)
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
        self, events_message: FileChangeEventsMessage
    ) -> None:
        """Add events to the in-memory rolling state. The next ``save_owner_state`` uploads it."""
        if self._rolling_state is None:
            return
        self._rolling_state.add_events_message(events_message)
        self._events_since_rolling_state_upload += len(events_message.events)

    def _upload_rolling_state_if_due(self) -> None:
        """Upload the rolling state once enough events were added since the last
        upload. A failed upload is logged, and the next ``save_owner_state`` tries again."""
        if (
            self._events_since_rolling_state_upload
            < DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD
        ):
            return
        try:
            self._upload_rolling_state()
        except Exception as e:
            logger.error(
                f"Rolling state upload failed, the next sync tries again: {e!r}"
            )

    def _upload_rolling_state(self) -> None:
        """Upload the in-memory rolling state to GDrive."""
        if self._rolling_state is None or self._rolling_state.event_count == 0:
            return

        self.connection_router.upload_rolling_state(self._rolling_state)
        self._events_since_rolling_state_upload = 0

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def _require_owner_state(self) -> None:
        """Refuse checkpoint work on a client that keeps no owner state."""
        if not self.persist_owner_state:
            raise ValueError(
                "Checkpoints are disabled: this client was built with "
                "persist_owner_state=False"
            )

    def create_incremental_checkpoint(self) -> IncrementalCheckpoint:
        """
        Create an incremental checkpoint from the current rolling state.

        Deduplicates events by path (keeps only latest version per file),
        then converts to an incremental checkpoint, uploads it,
        and deletes the rolling state.

        Returns:
            The created IncrementalCheckpoint object.
        """
        self._require_owner_state()
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
        self._require_owner_state()
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
        self._require_owner_state()
        print("Compacting incremental checkpoints...")

        # Download existing full checkpoint (if exists)
        existing_checkpoint = self.connection_router.get_latest_checkpoint()

        # Download all incremental checkpoints
        all_incremental_cps = self.connection_router.get_all_incremental_checkpoints()

        if not all_incremental_cps:
            raise ValueError("No incremental checkpoints to compact")
        incremental_cps = _incrementals_after(existing_checkpoint, all_incremental_cps)

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
        self._require_owner_state()
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
