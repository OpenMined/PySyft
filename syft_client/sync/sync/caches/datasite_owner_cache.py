from typing import List, Dict
from pydantic import ConfigDict, Field, model_validator
from syft_client.sync.events.file_change_event import FileChangeEventsMessage
from syft_client.sync.messages.proposed_filechange import ProposedFileChangesMessage
from uuid import uuid4
from pathlib import Path
from syft_client.sync.utils.syftbox_utils import create_event_timestamp
from syft_client.sync.messages.proposed_filechange import ProposedFileChange
from pydantic import BaseModel
from syft_client.sync.sync.caches.cache_file_writer_connection import FSFileConnection
from syft_client.sync.sync.caches.persisted_dict import PersistedDict
from syft_client.sync.events.file_change_event import FileChangeEvent
from syft_client.sync.callback_mixin import BaseModelCallbackMixin
from syft_client.sync.utils.path_filters import is_normal_syncable_path
from syft_client.sync.sync.caches.cache_file_writer_connection import (
    CacheFileConnection,
    InMemoryCacheFileConnection,
)
from syft_client.sync.utils.syftbox_utils import get_event_hash_from_content
from syft_client.sync.checkpoints.checkpoint import Checkpoint
from syft_client.sync.sync.constants import CACHE_DIR, OWNER_FILE_HASHES_FILENAME


class ProposedEventFileOutdatedException(Exception):
    def __init__(self, file_path: str, hash_in_event: int, hash_on_disk: int):
        super().__init__(
            f"Proposed event for file {file_path} is outdated, hash in event: {hash_in_event}, hash on disk: {hash_on_disk}"
        )


class DataSiteOwnerEventCacheConfig(BaseModel):
    use_in_memory_cache: bool = True
    syftbox_folder: Path | None = None
    email: str | None = None
    events_base_path: Path | None = None
    # Full path to collections folder - must be provided explicitly
    collections_folder: Path | None = None


class DataSiteOwnerEventCache(BaseModelCallbackMixin):
    # we keep a list of heads, which are the latest events for each path
    model_config = ConfigDict(arbitrary_types_allowed=True)

    events_messages_connection: CacheFileConnection = Field(
        default_factory=InMemoryCacheFileConnection
    )
    file_connection: CacheFileConnection = Field(
        default_factory=InMemoryCacheFileConnection
    )

    # When set, file_hashes auto-persists to {syftbox_folder}/CACHE_DIR/ OWNER_FILE_HASHES_FILENAME
    syftbox_folder: Path | None = None

    # file path to the hash of the filecontent.
    # Wired in pre-init validator: persisted-to-disk when syftbox_folder is set,
    # plain in-memory PersistedDict otherwise.
    file_hashes: PersistedDict = Field(default_factory=PersistedDict)
    email: str
    # Full path to collections (datasets) folder
    collections_folder: Path | None = None
    # Cache of collection hashes: "tag" -> content_hash
    collection_hashes: Dict[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def _build_file_hashes(cls, data):
        if isinstance(data, dict) and "file_hashes" not in data:
            folder = data.get("syftbox_folder")
            if folder is not None:
                data["file_hashes"] = PersistedDict(
                    path=Path(folder) / CACHE_DIR / OWNER_FILE_HASHES_FILENAME,
                    key_serializer=str,
                    key_deserializer=Path,
                )
        return data

    @classmethod
    def from_config(cls, config: DataSiteOwnerEventCacheConfig):
        if config.use_in_memory_cache:
            return cls(
                events_connection=InMemoryCacheFileConnection[FileChangeEvent](),
                file_connection=InMemoryCacheFileConnection[str](),
                email=config.email,
                syftbox_folder=config.syftbox_folder,
                file_hashes=PersistedDict(),
                collections_folder=config.collections_folder,
            )
        else:
            if config.syftbox_folder is None:
                raise ValueError("syftbox_folder is required for non-in-memory cache")
            if config.email is None:
                raise ValueError("email is required for non-in-memory cache")
            if config.collections_folder is None:
                raise ValueError(
                    "collections_folder is required for non-in-memory cache"
                )
            syftbox_folder_name = Path(config.syftbox_folder).name
            my_datasite_folder = config.syftbox_folder / config.email
            syftbox_parent = Path(config.syftbox_folder).parent
            events_folder = syftbox_parent / f"{syftbox_folder_name}-events"
            cache = cls(
                events_messages_connection=FSFileConnection(
                    base_dir=events_folder, dtype=FileChangeEventsMessage
                ),
                file_connection=FSFileConnection(base_dir=my_datasite_folder),
                syftbox_folder=config.syftbox_folder,
                email=config.email,
                collections_folder=config.collections_folder,
            )
            cache._load_cached_state()
            return cache

    def _load_cached_state(self):
        """Load cached state from disk: file hashes and collection hashes."""
        self._load_file_hashes_from_disk()
        self._load_collection_hashes_from_disk()

    def _load_file_hashes_from_disk(self) -> float | None:
        """Load existing events from disk and populate file_hashes.

        Hold a single exclusive lock for the entire replay and persist once
        at the end, instead of acquiring the lock and rewriting the cache
        file on every event.
        """
        cached_messages = self.events_messages_connection.get_all()

        sorted_messages = sorted(cached_messages, key=lambda m: m.timestamp)

        with self.file_hashes.exclusive_lock():
            for events_message in sorted_messages:
                for event in events_message.events:
                    if event.is_deleted:
                        if self.file_hashes.contains(
                            event.path_in_datasite, read=False
                        ):
                            self.file_hashes.delete(event.path_in_datasite, write=False)
                    else:
                        self.file_hashes.set(
                            event.path_in_datasite, event.new_hash, write=False
                        )
            self.file_hashes._write_to_file()

    def _load_collection_hashes_from_disk(self):
        """Scan local dataset directories and compute hashes to populate collection_hashes."""
        from syft_client.sync.file_utils import compute_directory_hash

        if self.collections_folder is None or not self.collections_folder.exists():
            return

        for tag_dir in self.collections_folder.iterdir():
            if tag_dir.is_dir():
                content_hash = compute_directory_hash(tag_dir)
                if content_hash:
                    self.collection_hashes[tag_dir.name] = content_hash

    def get_collection_hash(self, tag: str) -> str | None:
        """Get the cached hash for a collection."""
        return self.collection_hashes.get(tag)

    def set_collection_hash(self, tag: str, content_hash: str):
        """Set the cached hash for a collection."""
        self.collection_hashes[tag] = content_hash

    @property
    def latest_cached_timestamp(self) -> float | None:
        cached_messages = self.events_messages_connection.get_all()
        if not cached_messages:
            return None
        return max(m.timestamp for m in cached_messages)

    def collections_relative_path(self) -> Path:
        """Return the collections folder path relative to the datasite root."""
        return self.collections_folder.relative_to(self.syftbox_folder / self.email)

    def get_syncable_paths(self) -> dict[Path, bytes]:
        """Return {datasite-relative Path: content} for all normal-syncable files."""
        collections_rel = self.collections_relative_path()
        return {
            Path(path): content
            for path, content in self.file_connection.get_items()
            if is_normal_syncable_path(path, collections_path=collections_rel)
        }

    def process_local_file_changes(self) -> FileChangeEventsMessage | None:
        new_events = []

        current_files = self.get_syncable_paths()

        # Detect modifications and additions
        for path, content in current_files.items():
            current_hash = get_event_hash_from_content(content)
            if current_hash != self.file_hashes.get(path, None):
                timestamp = create_event_timestamp()
                event = FileChangeEvent(
                    id=uuid4(),
                    path_in_datasite=path,
                    content=content,
                    new_hash=current_hash,
                    old_hash=self.file_hashes.get(path),
                    submitted_timestamp=timestamp,
                    timestamp=timestamp,
                    datasite_email=self.email,
                    is_deleted=False,
                )
                new_events.append(event)

        # Detect deletions
        current_paths = set(current_files.keys())
        cached_paths = set(self.file_hashes.keys())
        deleted_paths = cached_paths - current_paths

        for deleted_path in deleted_paths:
            timestamp = create_event_timestamp()
            deletion_event = FileChangeEvent(
                id=uuid4(),
                path_in_datasite=deleted_path,
                content=None,
                old_hash=self.file_hashes[deleted_path],
                new_hash=None,
                submitted_timestamp=timestamp,
                timestamp=timestamp,
                datasite_email=self.email,
                is_deleted=True,
            )
            new_events.append(deletion_event)

        if new_events:
            events_message = FileChangeEventsMessage(events=new_events)
            # its already written so no need to write again
            self.add_events_message_to_local_cache(events_message, write_file=False)
            return events_message
        else:
            return None

    def create_events_for_files(
        self, files: dict[Path, bytes]
    ) -> FileChangeEventsMessage:
        """Create FileChangeEvents for a set of files and update the hash cache.

        Args:
            files: dict mapping path_in_datasite to file content bytes.

        Returns:
            FileChangeEventsMessage containing one event per file.
        """
        events = []
        for path_in_datasite, content in files.items():
            timestamp = create_event_timestamp()
            new_hash = get_event_hash_from_content(content)
            event = FileChangeEvent(
                id=uuid4(),
                path_in_datasite=path_in_datasite,
                content=content,
                old_hash=None,
                new_hash=new_hash,
                submitted_timestamp=timestamp,
                timestamp=timestamp,
                datasite_email=self.email,
                is_deleted=False,
            )
            events.append(event)
            self.file_hashes[path_in_datasite] = new_hash
        return FileChangeEventsMessage(events=events)

    def clear_cache(self):
        self.events_messages_connection.clear_cache()
        self.file_connection.clear_cache()
        self.file_hashes.clear()

    def has_conflict(self, proposed_event: ProposedFileChange) -> bool:
        if proposed_event.path_in_datasite not in self.file_hashes:
            if proposed_event.old_hash is None:
                return False
            else:
                raise ValueError(
                    f"File {proposed_event.path_in_datasite} is not in the cache but it does have an old hash"
                )
        return (
            self.file_hashes[proposed_event.path_in_datasite] != proposed_event.old_hash
        )

    def process_proposed_events_message(
        self, proposed_events_message: ProposedFileChangesMessage
    ) -> FileChangeEventsMessage | None:
        accepted_events_message = FileChangeEventsMessage(events=[])

        for proposed_filechange_event in proposed_events_message.proposed_file_changes:
            if self.has_conflict(proposed_filechange_event):
                hash_on_disk = self.file_hashes[
                    proposed_filechange_event.path_in_datasite
                ]
                raise ProposedEventFileOutdatedException(
                    proposed_filechange_event.path_in_datasite,
                    proposed_filechange_event.old_hash,
                    hash_on_disk,
                )
            else:
                accepted_event = FileChangeEvent.from_proposed_filechange(
                    proposed_filechange_event
                )
                accepted_events_message.events.append(accepted_event)
        if len(accepted_events_message.events) > 0:
            self.apply_accepted_events_message_to_cache(accepted_events_message)
            return accepted_events_message
        return None

    def apply_accepted_events_message_to_cache(
        self, accepted_events_message: FileChangeEventsMessage
    ):
        self.add_events_message_to_local_cache(accepted_events_message)

    def add_events_message_to_local_cache(
        self, accepted_events_message: FileChangeEventsMessage, write_file: bool = True
    ):
        self.events_messages_connection.write_file(
            path=accepted_events_message.message_filepath.as_string(),
            content=accepted_events_message,
        )

        for accepted_event in accepted_events_message.events:
            if accepted_event.is_deleted:
                # Handle deletion
                if accepted_event.path_in_datasite in self.file_hashes:
                    del self.file_hashes[accepted_event.path_in_datasite]

                if write_file:
                    self.file_connection.delete_file(accepted_event.path_in_datasite)

                for callback in self.callbacks.get("on_event_local_write", []):
                    callback(
                        accepted_event.path_in_datasite,
                        None,  # No content for deletions
                    )
            else:
                # Handle create/update
                self.file_hashes[accepted_event.path_in_datasite] = (
                    accepted_event.new_hash
                )

                if write_file:
                    self.file_connection.write_file(
                        accepted_event.path_in_datasite,
                        accepted_event.content,
                    )

                for callback in self.callbacks.get("on_event_local_write", []):
                    callback(
                        accepted_event.path_in_datasite,
                        accepted_event.content,
                    )

    def get_cached_events(self) -> List[FileChangeEvent]:
        events_messages = self.events_messages_connection.get_all()
        events = []
        for events_message in events_messages:
            events.extend(events_message.events)
        return events

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def create_checkpoint(
        self, last_event_timestamp: float | None = None
    ) -> Checkpoint:
        """
        Create a checkpoint from current cache state.

        Args:
            last_event_timestamp: Timestamp of the last event included in this checkpoint.

        Returns:
            A Checkpoint object containing all current files and their hashes.
        """
        file_contents = {
            str(path): content for path, content in self.get_syncable_paths().items()
        }

        return Checkpoint.from_file_hashes_and_contents(
            email=self.email,
            file_hashes=self.file_hashes,
            file_contents=file_contents,
            last_event_timestamp=last_event_timestamp,
        )

    def apply_checkpoint(self, checkpoint: Checkpoint, write_files: bool = True):
        """
        Restore cache state from a checkpoint.

        Args:
            checkpoint: The checkpoint to restore from.
            write_files: Whether to write files to disk (only affects filesystem,
                        in-memory file_connection is always updated).
        """
        # Clear current state
        self.file_hashes.clear()

        # Restore from checkpoint
        # Always write to file_connection to maintain consistent state for
        # process_local_changes comparison. This is needed even when write_files=False
        # because file_connection may be in-memory and serves as the source of truth.
        for file_entry in checkpoint.files:
            path = Path(file_entry.path)
            self.file_hashes[path] = file_entry.hash
            self.file_connection.write_file(file_entry.path, file_entry.content)

    def get_latest_event_timestamp(self) -> float | None:
        """Get the timestamp of the latest event in the cache."""
        events_messages = self.events_messages_connection.get_all()
        if not events_messages:
            return None

        latest_timestamp = None
        for events_message in events_messages:
            if latest_timestamp is None or events_message.timestamp > latest_timestamp:
                latest_timestamp = events_message.timestamp

        return latest_timestamp
