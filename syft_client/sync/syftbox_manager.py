from pathlib import Path
import fcntl
from syft_client.sync.peers.peer_store import PeerStore
from syft_client.sync.utils.path_filters import is_normal_syncable_path
import logging
import shutil
from contextlib import contextmanager
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.utils import resolve_path
from concurrent.futures import ThreadPoolExecutor
import time
from pydantic import ConfigDict
from syft_job.client import BaseJobClient, JobClient
from syft_job.job import JobsList
from syft_job.job_runner import SyftJobRunner
from syft_job import SyftJobConfig
from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_client.sync.platforms.base_platform import BasePlatform
from pydantic import BaseModel, PrivateAttr
from typing import List, Optional, cast
from syft_client.sync.sync.caches.datasite_watcher_cache import (
    DataSiteWatcherCacheConfig,
)
from syft_client.sync.sync.caches.datasite_owner_cache import (
    DataSiteOwnerEventCacheConfig,
)
from syft_client.sync.peers.peer_list import PeerList
from syft_client.sync.peers.peer import Peer
from syft_client.sync.connections.base_connection import (
    SyftboxPlatformConnection,
)
from syft_client.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
)
from syft_client.sync.utils.pre_submit_scan import run_pre_submit_check
from syft_client.sync.utils.syftbox_utils import (
    random_email,
    random_syftbox_folder_for_testing,
)
from syft_client.sync.file_writer import FileWriter

from syft_client.sync.job_file_change_handler import JobFileChangeHandler
from syft_client.sync.connections.connection_router import ConnectionRouter

from syft_client.sync.connections.drive.grdrive_config import GdriveConnectionConfig
from syft_client.sync.connections.drive import mock_drive_service
from syft_client.sync.sync.datasite_owner_syncer import (
    DatasiteOwnerSyncer,
    DatasiteOwnerSyncerConfig,
    MIN_MESSAGES_COMPACT,
)
from syft_client.sync.sync.datasite_watcher_syncer import (
    DatasiteWatcherSyncer,
    DatasiteWatcherSyncerConfig,
)
from syft_client.sync.version.peer_manager import (
    CompatAction,
    PeerManager,
    PeerManagerConfig,
)
from syft_client.sync.version.version_info import VersionInfo
from syft_client.version import VERSION_FILE_NAME
import os

logger = logging.getLogger(__name__)

COLAB_DEFAULT_SYFTBOX_FOLDER = Path("/")
JUPYTER_DEFAULT_SYFTBOX_FOLDER = Path.home() / "SyftBox"
COLLECTION_SUBPATH = Path("public/syft_datasets")

# ANSI codes for highlighting important warnings in terminals / notebooks.
_ANSI_RED = "\033[1;91m"
_ANSI_RESET = "\033[0m"


def get_jupyter_default_syftbox_folder(email: str):
    return Path.home() / f"SyftBox_{email}"


def get_colab_default_syftbox_folder(email: str):
    return Path("/content") / f"SyftBox_{email}"


class SyftboxManagerConfig(BaseModel):
    email: str
    syftbox_folder: Path
    write_files: bool = True
    has_ds_role: bool = False
    has_do_role: bool = False
    use_in_memory_cache: bool = True

    datasite_owner_syncer_config: DatasiteOwnerSyncerConfig
    peer_manager_config: PeerManagerConfig

    datasite_watcher_syncer_config: DatasiteWatcherSyncerConfig
    dataset_manager_config: SyftBoxConfig
    job_client_config: SyftJobConfig

    @classmethod
    def for_colab(
        cls,
        email: str,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        encryption: bool = False,
        encryption_keys: dict | None = None,
        skip_peer_on_patch_version_diff: Optional[
            bool
        ] = None,  # None: value is determined by the role
        force_ignore_peer_version: bool = False,
    ):
        if not has_ds_role and not has_do_role:
            raise ValueError("At least one of has_ds_role or has_do_role must be True")

        syftbox_folder = get_colab_default_syftbox_folder(email)
        use_in_memory_cache = False
        collections_folder = syftbox_folder / email / COLLECTION_SUBPATH
        connection_configs = [GdriveConnectionConfig(email=email, token_path=None)]
        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            connection_configs=connection_configs,
            cache_config=DataSiteOwnerEventCacheConfig(
                email=email,
                use_in_memory_cache=use_in_memory_cache,
                syftbox_folder=syftbox_folder,
                collections_folder=collections_folder,
            ),
        )
        datasite_watcher_syncer_config = DatasiteWatcherSyncerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
            connection_configs=connection_configs,
            datasite_watcher_cache_config=DataSiteWatcherCacheConfig(
                email=email,
                use_in_memory_cache=use_in_memory_cache,
                syftbox_folder=syftbox_folder,
                collection_subpath=COLLECTION_SUBPATH,
                connection_configs=connection_configs,
            ),
        )
        job_client_config = SyftJobConfig(
            syftbox_folder=syftbox_folder,
            current_user_email=email,
            has_do_role=has_do_role,
        )
        dataset_manager_config = SyftBoxConfig(
            syftbox_folder=syftbox_folder,
            email=email,
        )
        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            connection_configs=connection_configs,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            use_encryption=encryption,
            encryption_keys=encryption_keys,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
            force_ignore_peer_version=force_ignore_peer_version,
        )
        return cls(
            email=email,
            syftbox_folder=syftbox_folder,
            has_ds_role=has_ds_role,
            has_do_role=has_do_role,
            connection_configs=connection_configs,
            use_in_memory_cache=False,
            datasite_owner_syncer_config=datasite_owner_syncer_config,
            datasite_watcher_syncer_config=datasite_watcher_syncer_config,
            dataset_manager_config=dataset_manager_config,
            job_client_config=job_client_config,
            peer_manager_config=peer_manager_config,
        )

    @classmethod
    def for_jupyter(
        cls,
        email: str,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        token_path: Path | None = None,
        encryption: bool = False,
        encryption_keys: dict | None = None,
        skip_peer_on_patch_version_diff: Optional[
            bool
        ] = None,  # None: value is determined by the role
        force_ignore_peer_version: bool = False,
    ):
        if not has_ds_role and not has_do_role:
            raise ValueError("At least one of has_ds_role or has_do_role must be True")

        syftbox_folder = get_jupyter_default_syftbox_folder(email)
        collections_folder = syftbox_folder / email / COLLECTION_SUBPATH

        connection_configs = [
            GdriveConnectionConfig(email=email, token_path=token_path)
        ]
        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            connection_configs=connection_configs,
            cache_config=DataSiteOwnerEventCacheConfig(
                email=email,
                use_in_memory_cache=False,
                syftbox_folder=syftbox_folder,
                collections_folder=collections_folder,
                connection_configs=connection_configs,
            ),
        )
        datasite_watcher_syncer_config = DatasiteWatcherSyncerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
            connection_configs=connection_configs,
            datasite_watcher_cache_config=DataSiteWatcherCacheConfig(
                email=email,
                use_in_memory_cache=False,
                syftbox_folder=syftbox_folder,
                collection_subpath=COLLECTION_SUBPATH,
                connection_configs=connection_configs,
            ),
        )
        dataset_manager_config = SyftBoxConfig(
            syftbox_folder=syftbox_folder,
            email=email,
        )
        job_client_config = SyftJobConfig(
            syftbox_folder=syftbox_folder,
            current_user_email=email,
            has_do_role=has_do_role,
        )
        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            connection_configs=connection_configs,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            use_encryption=encryption,
            encryption_keys=encryption_keys,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
            force_ignore_peer_version=force_ignore_peer_version,
        )
        return cls(
            email=email,
            syftbox_folder=syftbox_folder,
            has_ds_role=has_ds_role,
            has_do_role=has_do_role,
            use_in_memory_cache=False,
            datasite_owner_syncer_config=datasite_owner_syncer_config,
            datasite_watcher_syncer_config=datasite_watcher_syncer_config,
            dataset_manager_config=dataset_manager_config,
            job_client_config=job_client_config,
            peer_manager_config=peer_manager_config,
        )

    @classmethod
    def _base_config_for_testing(
        cls,
        email: str | None = None,
        syftbox_folder: Path | None = None,
        write_files: bool = False,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        use_in_memory_cache: bool = True,
        check_versions: bool = False,
    ):
        syftbox_folder = syftbox_folder or random_syftbox_folder_for_testing()
        email = email or random_email()
        collections_folder = syftbox_folder / email / COLLECTION_SUBPATH

        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            write_files=write_files,
            cache_config=DataSiteOwnerEventCacheConfig(
                email=email,
                use_in_memory_cache=use_in_memory_cache,
                syftbox_folder=syftbox_folder,
                collections_folder=collections_folder,
            ),
        )
        datasite_watcher_syncer_config = DatasiteWatcherSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            datasite_watcher_cache_config=DataSiteWatcherCacheConfig(
                email=email,
                use_in_memory_cache=use_in_memory_cache,
                syftbox_folder=syftbox_folder,
                collection_subpath=COLLECTION_SUBPATH,
            ),
        )

        dataset_manager_config = SyftBoxConfig(
            syftbox_folder=syftbox_folder,
            email=email,
        )
        job_client_config = SyftJobConfig(
            syftbox_folder=Path(syftbox_folder),
            current_user_email=email,
            has_do_role=has_do_role,
        )
        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            connection_configs=[],  # Empty for in-memory, connections added later
            n_threads=2,  # Use fewer threads for testing
            force_ignore_peer_version=not check_versions,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
        )

        return cls(
            email=email,
            syftbox_folder=syftbox_folder,
            write_files=write_files,
            has_ds_role=has_ds_role,
            has_do_role=has_do_role,
            use_in_memory_cache=use_in_memory_cache,
            datasite_owner_syncer_config=datasite_owner_syncer_config,
            datasite_watcher_syncer_config=datasite_watcher_syncer_config,
            dataset_manager_config=dataset_manager_config,
            job_client_config=job_client_config,
            peer_manager_config=peer_manager_config,
        )

    @classmethod
    def for_google_drive_testing_connection(
        cls,
        email: str,
        token_path: Path,
        syftbox_folder: str | None = None,
        write_files: bool = False,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        use_in_memory_cache: bool = True,
        check_versions: bool = False,
    ):
        syftbox_folder = syftbox_folder or random_syftbox_folder_for_testing()
        email = email or random_email()
        collections_folder = Path(syftbox_folder) / email / COLLECTION_SUBPATH
        connection_configs = [
            GdriveConnectionConfig(email=email, token_path=token_path)
        ]
        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            connection_configs=connection_configs,
            cache_config=DataSiteOwnerEventCacheConfig(
                email=email,
                use_in_memory_cache=use_in_memory_cache,
                syftbox_folder=syftbox_folder,
                collections_folder=collections_folder,
            ),
        )
        datasite_watcher_syncer_config = DatasiteWatcherSyncerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
            connection_configs=connection_configs,
            datasite_watcher_cache_config=DataSiteWatcherCacheConfig(
                email=email,
                use_in_memory_cache=use_in_memory_cache,
                syftbox_folder=syftbox_folder,
                collection_subpath=COLLECTION_SUBPATH,
                connection_configs=connection_configs,
            ),
        )

        dataset_manager_config = SyftBoxConfig(
            syftbox_folder=syftbox_folder,
            email=email,
        )
        job_client_config = SyftJobConfig(
            syftbox_folder=syftbox_folder,
            current_user_email=email,
            has_do_role=has_do_role,
        )
        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            connection_configs=connection_configs,
            force_ignore_peer_version=not check_versions,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
        )
        return cls(
            email=email,
            syftbox_folder=syftbox_folder,
            write_files=write_files,
            datasite_owner_syncer_config=datasite_owner_syncer_config,
            datasite_watcher_syncer_config=datasite_watcher_syncer_config,
            has_ds_role=has_ds_role,
            has_do_role=has_do_role,
            use_in_memory_cache=False,
            dataset_manager_config=dataset_manager_config,
            job_client_config=job_client_config,
            peer_manager_config=peer_manager_config,
        )


class SyftboxManager(BaseModel):
    # needed for peers
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_writer: FileWriter
    syftbox_folder: Path
    email: str
    dev_mode: bool = False
    datasite_watcher_syncer: DatasiteWatcherSyncer | None = None

    datasite_owner_syncer: DatasiteOwnerSyncer | None = None
    job_file_change_handler: JobFileChangeHandler | None = None
    dataset_manager: SyftDatasetManager | None = None
    job_client: BaseJobClient | None = None
    job_runner: SyftJobRunner | None = None
    peer_manager: PeerManager | None = None
    has_do_role: bool = False
    has_ds_role: bool = False
    config: SyftboxManagerConfig | None = None

    _executor: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=10)
    )
    _peer_store: object = PrivateAttr(default=None)

    _PUBLIC_API = (
        "email",
        "syftbox_folder",
        "dev_mode",
        "config",
        "has_do_role",
        "has_ds_role",
        "dataset_manager",
        "peer_manager",
        "peers",
        "jobs",
        "datasets",
        "add_peer",
        "load_peers",
        "approve_peer_request",
        "reject_peer_request",
        "sync",
        "create_dataset",
        "delete_dataset",
        "share_dataset",
        "submit_bash_job",
        "submit_python_job",
        "process_approved_jobs",
        "create_checkpoint",
        "should_create_checkpoint",
        "try_create_checkpoint",
        "delete_syftbox",
        "write_own_version",
    )

    def __dir__(self):
        return list(self._PUBLIC_API)

    def read_local_version(self) -> VersionInfo | None:
        """Read the local SYFT_version.json from the SyftBox directory."""
        version_file = self.syftbox_folder / VERSION_FILE_NAME
        if not version_file.exists():
            return None
        try:
            return VersionInfo.from_json(version_file.read_text())
        except Exception:
            return None

    def write_local_version(self) -> None:
        """Write current version info to a local SYFT_version.json."""
        self.syftbox_folder.mkdir(parents=True, exist_ok=True)
        version_file = self.syftbox_folder / VERSION_FILE_NAME
        version_file.write_text(VersionInfo.current().to_json())

    @property
    def peers(self) -> PeerList:
        """
        Get the combined list of peers (approved + requests).
        Automatically calls sync() before returning peers
        if PRE_SYNC environment variable is set to "true" (case-insensitive).

        PRE_SYNC defaults to "true", so auto-sync is enabled by default.
        To disable auto-sync, set: PRE_SYNC=false

        Returns PeerList with approved peers first, then requests.
        """
        if os.environ.get("PRE_SYNC", "true").lower() == "true":
            self.sync()

        vm = self.peer_manager
        combined = PeerList(
            vm.approved_peers + vm.requested_by_me_peers + vm.requested_by_peer_peers
        )
        for peer in combined:
            peer._manager = self
        return combined

    @classmethod
    def from_config(cls, config: SyftboxManagerConfig):
        file_writer = FileWriter(
            base_path=config.syftbox_folder, write_files=config.write_files
        )

        datasite_owner_syncer = None
        job_file_change_handler = None
        datasite_watcher_syncer = None
        job_runner = None

        dataset_manager = SyftDatasetManager.from_config(config.dataset_manager_config)
        job_client = JobClient.from_config(config.job_client_config)

        if config.has_do_role:
            datasite_owner_syncer = DatasiteOwnerSyncer.from_config(
                config.datasite_owner_syncer_config
            )

            job_file_change_handler = JobFileChangeHandler()
            job_runner = SyftJobRunner.from_config(config.job_client_config)

        if config.has_ds_role:
            datasite_watcher_syncer = DatasiteWatcherSyncer.from_config(
                config.datasite_watcher_syncer_config
            )

        peer_manager = PeerManager.from_config(
            config.peer_manager_config, email=config.email
        )

        manager_res = cls(
            syftbox_folder=config.syftbox_folder,
            email=config.email,
            file_writer=file_writer,
            datasite_owner_syncer=datasite_owner_syncer,
            job_file_change_handler=job_file_change_handler,
            datasite_watcher_syncer=datasite_watcher_syncer,
            dataset_manager=dataset_manager,
            job_client=job_client,
            job_runner=job_runner,
            peer_manager=peer_manager,
            has_do_role=config.has_do_role,
            has_ds_role=config.has_ds_role,
            config=config,
        )

        # PeerManager.from_config built the (possibly key-bearing) peer store from
        # peer_manager_config; share that single store across every syncer's router
        # so encryption is consistent on all sync paths.
        if peer_manager.peer_store.use_encryption:
            manager_res._set_peer_store(peer_manager.peer_store)

        return manager_res

    def _set_peer_store(self, peer_store) -> None:
        """Wire shared peer_store into all connection routers."""
        from syft_client.sync.peers.peer_store import PeerStore

        if not isinstance(peer_store, PeerStore):
            return
        self._peer_store = peer_store
        if self.peer_manager:
            self.peer_manager.peer_store = peer_store
            self.peer_manager.connection_router.peer_store = peer_store
        if self.datasite_owner_syncer:
            self.datasite_owner_syncer.connection_router.peer_store = peer_store
        if self.datasite_watcher_syncer:
            self.datasite_watcher_syncer.connection_router.peer_store = peer_store
            self.datasite_watcher_syncer.datasite_watcher_cache.connection_router.peer_store = peer_store

    @classmethod
    def for_colab(
        cls,
        email: str,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        encryption: bool = False,
        encryption_keys: dict | None = None,
        skip_peer_on_patch_version_diff: Optional[
            bool
        ] = None,  # None: value is determined by the role
        force_ignore_peer_version: bool = False,
    ):
        manager = cls.from_config(
            SyftboxManagerConfig.for_colab(
                email=email,
                has_ds_role=has_ds_role,
                has_do_role=has_do_role,
                encryption=encryption,
                encryption_keys=encryption_keys,
                skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
                force_ignore_peer_version=force_ignore_peer_version,
            )
        )
        return manager

    def _init_encryption(self, encryption_keys: dict | None = None) -> None:
        """Overwrite this manager's encryption keys from the top level.

        Standard key init happens lower down in ``PeerManager.from_config`` via
        the config. Call this to swap in a different key bundle (or freshly
        generated/persisted keys when omitted) after construction, wiring the
        new store into every connection router.
        """
        from syft_client.sync.peers.peer_store import PeerStore

        ps = PeerStore.create(
            email=self.email, use_encryption=True, encryption_keys=encryption_keys
        )
        self._set_peer_store(ps)

    @classmethod
    def for_jupyter(
        cls,
        email: str,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        token_path: Path | None = None,
        encryption: bool = False,
        encryption_keys: dict | None = None,
        skip_peer_on_patch_version_diff: Optional[
            bool
        ] = None,  # None: value is determined by the role
        force_ignore_peer_version: bool = False,
    ):
        if token_path is not None:
            token_path = Path(token_path)
        manager = cls.from_config(
            SyftboxManagerConfig.for_jupyter(
                email=email,
                has_ds_role=has_ds_role,
                has_do_role=has_do_role,
                token_path=token_path,
                encryption=encryption,
                encryption_keys=encryption_keys,
                skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
                force_ignore_peer_version=force_ignore_peer_version,
            )
        )
        return manager

    @classmethod
    def _pair_with_google_drive_testing_connection(
        cls,
        do_email: str,
        ds_email: str,
        do_token_path: Path,
        ds_token_path: Path,
        base_path1: str | None = None,
        base_path2: str | None = None,
        add_peers: bool = True,
        load_peers: bool = False,
        use_in_memory_cache: bool = True,
        clear_caches: bool = True,
        check_versions: bool = False,
    ):
        receiver_config = SyftboxManagerConfig.for_google_drive_testing_connection(
            email=do_email,
            syftbox_folder=base_path1,
            use_in_memory_cache=use_in_memory_cache,
            token_path=do_token_path,
            has_ds_role=False,
            has_do_role=True,
            check_versions=check_versions,
        )

        receiver_manager = cls.from_config(receiver_config)

        sender_config = SyftboxManagerConfig.for_google_drive_testing_connection(
            email=ds_email,
            syftbox_folder=base_path2,
            use_in_memory_cache=use_in_memory_cache,
            token_path=ds_token_path,
            has_ds_role=True,
            has_do_role=False,
            check_versions=check_versions,
        )
        sender_manager = cls.from_config(sender_config)

        # Write version files if version checking is enabled
        if check_versions:
            sender_manager.peer_manager.write_own_version()
            receiver_manager.peer_manager.write_own_version()

        # this makes sure that when we write a file as sender, the inactive file watcher picks it up
        sender_manager.file_writer.add_callback(
            "write_file",
            sender_manager.datasite_watcher_syncer.on_file_change,
        )

        # this makes sure that when we receive a message, the handler is called
        # receiver_manager.proposed_file_change_puller.add_callback(
        #     "on_proposed_filechange_receive",
        #     receiver_manager.datasite_owner_syncer.handle_proposed_filechange_event,
        # )
        # this make sure that when the receiver writes a file to disk,
        # the file watcher picks it up
        # we use the underscored method to allow for monkey patching
        receiver_manager.datasite_owner_syncer.event_cache.add_callback(
            "on_event_local_write",
            receiver_manager.job_file_change_handler._handle_file_change,
        )

        if clear_caches:
            receiver_manager._clear_caches()
            sender_manager._clear_caches()

        if add_peers:
            # DS creates peer request
            sender_manager.add_peer(receiver_manager.email, sync=False)
            # unfortunately, we need this because of delays in gdrive
            # DO approves the peer request automatically (for backward compatibility)
            receiver_manager.load_peers()
            # we are not checking if the peer exists because of delays in gdrive
            receiver_manager.approve_peer_request(
                sender_manager.email, peer_must_exist=False
            )
        if load_peers:
            receiver_manager.load_peers()
            sender_manager.load_peers()

        # create inbox folder
        return sender_manager, receiver_manager

    @classmethod
    def pair_with_mock_drive_service_connection(
        cls,
        email1: str | None = None,
        email2: str | None = None,
        base_path1: str | None = None,
        base_path2: str | None = None,
        sync_automatically: bool = False,
        add_peers: bool = True,
        use_in_memory_cache: bool = True,
        check_versions: bool = False,
        encryption: bool = False,
    ):
        """Create a pair of managers using mock Google Drive services for testing.

        This creates managers that use the actual GDriveConnection code but with
        mock services instead of real Google Drive API calls. This allows testing
        the full GDrive code path without network calls.

        Args:
            email1: Email for the DO manager (defaults to random)
            email2: Email for the DS manager (defaults to random)
            base_path1: Base path for DO manager (defaults to temp dir)
            base_path2: Base path for DS manager (defaults to temp dir)
            sync_automatically: Whether to sync when DS sends changes
            add_peers: Whether to automatically add and approve peers
            use_in_memory_cache: Whether to use in-memory caches
            check_versions: Whether to check protocol/client versions

        Returns:
            Tuple of (ds_manager, do_manager)
        """
        # Create configs using the existing base config generator
        do_config = SyftboxManagerConfig._base_config_for_testing(
            email=email1,
            syftbox_folder=base_path1,
            has_ds_role=False,
            has_do_role=True,
            use_in_memory_cache=use_in_memory_cache,
            check_versions=check_versions,
        )

        ds_config = SyftboxManagerConfig._base_config_for_testing(
            email=email2,
            syftbox_folder=base_path2,
            has_ds_role=True,
            has_do_role=False,
            use_in_memory_cache=use_in_memory_cache,
            check_versions=check_versions,
        )

        # Create managers from configs
        do_manager = cls.from_config(do_config)
        ds_manager = cls.from_config(ds_config)

        # Create GDriveConnection instances with mock services
        do_connection, ds_connection = mock_drive_service.pair_with_mock_service(
            do_manager.email, ds_manager.email
        )

        # Add connections to managers
        do_manager._add_connection(do_connection)
        ds_manager._add_connection(ds_connection)

        # Set up callbacks for DS -> DO communication
        ds_manager.file_writer.add_callback(
            "write_file",
            ds_manager.datasite_watcher_syncer.on_file_change,
        )

        # Set up callback for DO job handling
        do_manager.datasite_owner_syncer.event_cache.add_callback(
            "on_event_local_write",
            do_manager.job_file_change_handler._handle_file_change,
        )

        # Write version files
        ds_manager.peer_manager.write_own_version()
        do_manager.peer_manager.write_own_version()

        # Initialize encryption if requested
        if encryption:
            ds_manager._init_encrypted_peer_store()
            do_manager._init_encrypted_peer_store()

        if add_peers:
            # DS creates peer request
            ds_manager.add_peer(do_manager.email, sync=False)
            # DO approves the peer request
            do_manager.load_peers()
            do_manager.approve_peer_request(ds_manager.email)
            # DS refreshes peer state so it sees DO's accepted status and
            # DO's advertised VersionInfo (incl. syft_client_install_source).
            ds_manager.load_peers()

        return ds_manager, do_manager

    def _init_encrypted_peer_store(self) -> None:
        """Initialize the encrypted peer store."""
        peer_store = PeerStore(email=self.email, use_encryption=True)
        peer_store.generate_keys()
        self._set_peer_store(peer_store)

    def add_peer(
        self,
        peer_email: str,
        force: bool = False,
        verbose: bool = True,
        sync: bool = True,
    ):
        """Add a peer. Delegates to PeerManager."""
        self.peer_manager.add_peer(peer_email, force=force, verbose=verbose)
        self._sync_peer_install_sources_to_job_client()
        if self.has_do_role:
            self._post_approve_peer_do(peer_email)
        if sync:
            self.sync()

    def submit_bash_job(
        self,
        user: str,
        script: str,
        job_name: str = "",
        sync=True,
        force_submission: bool = False,
        ignore_peer_version: bool = False,
    ):
        # Check version compatibility before submission (uses cached versions)
        if not force_submission:
            result = self.peer_manager.get_peer_compatibility_status(
                user,
                action=CompatAction.SUBMIT,
                ignore_peer_version=ignore_peer_version,
            )
            result.raise_on_skip(operation="submit job")
            result.maybe_warn()
        job_dir = self.job_client.submit_bash_job(user, script, job_name=job_name)
        self.push_job_files(job_dir)

    def submit_python_job(
        self,
        user: str,
        code_path: str,
        job_name: str | None = "",
        dependencies: list[str] | None = None,
        entrypoint: str | None = None,
        sync=True,
        force_submission: bool = False,
        ignore_peer_version: bool = False,
    ):
        peer_emails = {p.email for p in self.peer_manager.syncable_peers}
        if user not in peer_emails:
            print(f"⚠️  {user} is not in your peer list.")
            print(f"   Add them first with: client.add_peer('{user}')")
            return

        if not force_submission:
            if not run_pre_submit_check(Path(code_path)):
                print("Submission aborted.")
                return

        print(f"📤 Submitting '{code_path}' to {user}...")
        if job_name:
            print(f"   Job name     : {job_name}")
        if dependencies:
            print(f"   Dependencies : {', '.join(dependencies)}")

        # Check version compatibility before submission (uses cached versions)
        if not force_submission:
            result = self.peer_manager.get_peer_compatibility_status(
                user,
                action=CompatAction.SUBMIT,
                ignore_peer_version=ignore_peer_version,
            )
            result.raise_on_skip(operation="submit job")
            result.maybe_warn()
        job_dir = self.job_client.submit_python_job(
            user,
            code_path,
            job_name=job_name,
            dependencies=dependencies,
            entrypoint=entrypoint,
        )
        self.push_job_files(job_dir)

        print("\n✅ Job submitted successfully!")
        print("   Status : inbox (waiting for DO to review)")
        print(f"\n⏳ Next step: wait for {user} to approve and run it.")
        print("   Check progress with: client.jobs")

    def push_job_files(self, job_dir: Path):
        file_paths = [Path(p) for p in job_dir.rglob("*") if p.is_file()]
        relative_file_paths = [p.relative_to(self.syftbox_folder) for p in file_paths]
        for rel in relative_file_paths:
            # job_dir lives under <syftbox>/<owning_email>/...; strip the email
            # component to get a datasite-relative path for the filter check.
            datasite_rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
            if not is_normal_syncable_path(datasite_rel):
                logger.warning(f"push_job_files: pushing non-syncable path {rel}")

        last_file = False
        for i, relative_file_path in enumerate(relative_file_paths):
            # only send a message for the last file, so we reduce the number of messages sent
            if i == len(relative_file_paths) - 1:
                last_file = True

            self.datasite_watcher_syncer.on_file_change(
                relative_file_path, process_now=last_file
            )

    @contextmanager
    def _sync_file_lock(self):
        lock_path = self.syftbox_folder / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
        with open(lock_path, "r") as lock_handle:
            try:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    logger.info(f"Waiting for sync lock for {self.email}...")
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    logger.info(f"Sync lock acquired for {self.email}")
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def sync(
        self,
        auto_checkpoint: bool = True,
        checkpoint_threshold: int = 50,
        auto_compact: bool = True,
        compact_threshold: int = MIN_MESSAGES_COMPACT,
        force_download_peer_state: bool = False,
        ignore_peer_version: bool = False,
    ):
        """
        Sync local state with Google Drive.

        Args:
            auto_checkpoint: If True, automatically create checkpoint when
                            event count exceeds threshold (DO only).
            checkpoint_threshold: Create checkpoint when events >= this value.
            auto_compact: If True, after each DO sync, compact each peer's
                          outbox if it holds at least `compact_threshold`
                          message files (DO only).
            compact_threshold: Compact a peer's outbox when message-file
                          count >= this value.
            force_download_peer_state: If True, re-fetch SYFT_peers.json
                          from Drive instead of using the cached copy. Use
                          from background daemons (e.g. syft-bg) where a
                          separate process may have updated the file.
        """
        with self._sync_file_lock():
            self.load_peers(force_download=force_download_peer_state)
            if self.has_do_role:
                peer_emails = [peer.email for peer in self.peer_manager.approved_peers]
                compatible_emails = (
                    self.peer_manager.get_compatible_peer_emails_for_syncing(
                        peer_emails,
                        ignore_peer_version=ignore_peer_version,
                    )
                )
                self.datasite_owner_syncer.sync(compatible_emails)
                if auto_compact:
                    for peer_email in compatible_emails:
                        self.datasite_owner_syncer.compact_outbox_if_needed(
                            peer_email, min_messages=compact_threshold
                        )
                if auto_checkpoint:
                    self.try_create_checkpoint(checkpoint_threshold)

            if self.has_ds_role:
                peer_emails = [peer.email for peer in self.peer_manager.syncable_peers]
                self.peer_manager.warn_if_all_peers_incompatible(peer_emails)
                self.datasite_watcher_syncer.sync_down(peer_emails)

    def load_peers(self, force_download: bool = False):
        """Load peers from connection router. Delegates to PeerManager.

        Args:
            force_download: If True, re-fetch SYFT_peers.json from Drive
                instead of using the cached copy.
        """
        cast(PeerManager, self.peer_manager).load_peers(force_download=force_download)
        self._sync_peer_install_sources_to_job_client()

    def _check_peer_request_exists(self, email: str) -> bool:
        """Check if a peer request exists. Delegates to PeerManager."""
        return self.peer_manager.check_peer_request_exists(email)

    def approve_peer_request(
        self,
        email_or_peer: str | Peer,
        verbose: bool = True,
        peer_must_exist: bool = True,
    ):
        """Approve a pending peer request. Delegates to PeerManager."""
        self.peer_manager.approve_peer_request(
            email_or_peer, verbose=verbose, peer_must_exist=peer_must_exist
        )
        self._sync_peer_install_sources_to_job_client()
        self._post_approve_peer_do(email_or_peer)

    def _post_approve_peer_do(self, email_or_peer: str | Peer):
        """Shared post-approval logic: job folder setup and dataset sharing."""
        peer_email = (
            email_or_peer if isinstance(email_or_peer, str) else email_or_peer.email
        )

        if self.has_do_role:
            self.job_client.setup_ds_job_folder_as_do(peer_email)
            self._share_any_datasets_with_peer(peer_email)

    def _sync_peer_install_sources_to_job_client(self) -> None:
        """Copy each peer's advertised syft-client install source into job_client.

        Called after peer version exchanges so that, when the DS submits a job
        to a DO, the run.sh references the DO's local install path rather than
        the DS's local detection.
        """
        if not self.job_client:
            return
        for peer in self.peer_manager.peer_store.syncable_peers:
            source = peer.version.syft_client_install_source if peer.version else None
            if source:
                self.job_client.peer_install_sources[peer.email] = source

    def _ensure_local_peer_permissions(self) -> None:
        """Recreate local permission files for all approved peers.

        After an upgrade that deletes local state, permission files
        (syft.pub.yaml) are lost. This restores them so that incoming
        proposals from approved peers are not silently rejected.
        """
        if not self.has_do_role:
            return
        for peer in self.peer_manager.approved_peers:
            self.job_client.setup_ds_job_folder_as_do(peer.email)

    def _share_any_datasets_with_peer(self, peer_email: str):
        """Share all datasets that have 'any' permission with a specific peer.

        This is needed because Google Drive "anyone with link" files are not
        discoverable via search. By adding explicit user sharing, the peer
        can discover these datasets.

        Uses cache populated during pull_initial_state() in DatasiteOwnerSyncer.
        """
        for tag, content_hash in self.datasite_owner_syncer._any_shared_datasets:
            try:
                self._connection_router.owner_share_dataset_collection(
                    tag, content_hash, [peer_email]
                )
            except Exception:
                # Ignore errors (e.g., already shared)
                pass

    def reject_peer_request(self, email_or_peer: str | Peer):
        """Reject a pending peer request. Delegates to PeerManager."""
        self.peer_manager.reject_peer_request(email_or_peer)

    @property
    def jobs(self) -> JobsList:
        """
        Get the list of jobs. Automatically calls sync() before returning jobs
        if PRE_SYNC environment variable is set to "true" (case-insensitive).

        PRE_SYNC defaults to "true", so auto-sync is enabled by default.
        To disable auto-sync, set: PRE_SYNC=false
        """
        if os.environ.get("PRE_SYNC", "true").lower() == "true":
            self.sync()
        return self.job_client.jobs

    def process_approved_jobs(
        self,
        stream_output: bool = True,
        timeout: int | None = None,
        force_execution: bool = False,
        share_outputs_with_submitter: bool = False,
        share_logs_with_submitter: bool = False,
        ignore_peer_version: bool = False,
    ) -> None:
        """
        Process approved jobs. Automatically calls sync() after processing

        Args:
            stream_output: If True (default), stream output in real-time.
                        If False, capture output at end.
            timeout: Timeout in seconds per job. Defaults to 300 (5 minutes).
                    Can also be set via SYFT_DEFAULT_JOB_TIMEOUT_SECONDS env var.
            force_execution: If True, process all approved jobs regardless of
                           version compatibility. If False (default), skip jobs
                           from peers with incompatible or unknown versions.
            share_outputs_with_submitter: If True, grant read access on outputs to submitter.
            share_logs_with_submitter: If True, grant read access on logs to submitter.

        PRE_SYNC defaults to "true", so auto-sync is enabled by default.
        To disable auto-sync, set: PRE_SYNC=false
        """
        skip_job_names = []

        if not force_execution:
            approved_jobs = [
                job for job in self.job_client.jobs if job.status == "approved"
            ]
            for job in approved_jobs:
                result = self.peer_manager.get_peer_compatibility_status(
                    job.submitted_by,
                    action=CompatAction.EXECUTE,
                    ignore_peer_version=ignore_peer_version,
                )
                result.maybe_warn()
                if result.should_skip:
                    skip_job_names.append(job.name)

        self.job_runner.process_approved_jobs(
            stream_output=stream_output,
            timeout=timeout,
            skip_job_names=skip_job_names if skip_job_names else None,
            share_outputs_with_submitter=share_outputs_with_submitter,
            share_logs_with_submitter=share_logs_with_submitter,
        )

        if os.environ.get("PRE_SYNC", "true").lower() == "true":
            self.sync()

    def _add_connection(self, connection: SyftboxPlatformConnection):
        if not (
            isinstance(connection, GDriveConnection)
            and isinstance(
                connection.drive_service, mock_drive_service.MockDriveService
            )
        ):
            raise ValueError(
                "Only MockDriveService connections can be added to the manager"
            )

        if self.datasite_owner_syncer is not None:
            self.datasite_owner_syncer.connection_router.add_connection(connection)
        if self.datasite_watcher_syncer is not None:
            self.datasite_watcher_syncer.connection_router.add_connection(connection)
            self.datasite_watcher_syncer.datasite_watcher_cache.connection_router.add_connection(
                connection
            )

        # Add connection to version manager's router
        self.peer_manager.connection_router.add_connection(connection)

    def _send_file_change(self, path: str | Path, content: str):
        self.file_writer.write_file(path, content)

    def _get_all_accepted_events_do(self) -> List[FileChangeEvent]:
        return self.datasite_owner_syncer.connection_router.owner_get_all_accepted_events_messages()

    def create_dataset(
        self,
        name: str,
        mock_path: str | Path,
        private_path: str | Path,
        summary: str | None = None,
        readme_path: Path | None = None,
        location: str | None = None,
        tags: list[str] | None = None,
        users: list[str] | str | None = None,
        upload_private: bool = False,
        sync=True,
    ):
        if self.dataset_manager is None:
            raise ValueError("Dataset manager is not set")

        # Only DO can create datasets
        if not self.has_do_role:
            raise ValueError("Only dataset owners can create datasets")

        # Convert None to empty list
        if users is None:
            users = []

        dataset_name = None
        created_local = False
        mock_folder_id = None
        private_folder_id = None

        try:
            # Create dataset locally
            dataset = self.dataset_manager.create(
                name=name,
                mock_path=mock_path,
                private_path=private_path,
                summary=summary,
                readme_path=readme_path,
                location=location,
                tags=tags,
                users=users,
            )
            created_local = True
            dataset_name = dataset.name

            # Upload mock data to collection folder
            mock_folder_id = self._upload_dataset_to_collection(dataset, users)

            # Upload private data to a separate owner-only collection
            if upload_private:
                private_folder_id = self._upload_private_dataset_to_collection(dataset)

            if sync:
                self.sync()

            return dataset

        except Exception:
            logger.error(
                "Failed to create dataset%s, cleaning up",
                f" '{dataset_name}'" if dataset_name else "",
            )
            self._cleanup_failed_dataset_creation(
                dataset_name, created_local, mock_folder_id, private_folder_id
            )
            raise

    def _cleanup_failed_dataset_creation(
        self,
        dataset_name: str | None,
        created_local: bool,
        mock_folder_id: str | None,
        private_folder_id: str | None,
    ) -> None:
        """Best-effort cleanup after a failed create_dataset, in reverse order."""
        if private_folder_id is not None:
            try:
                self._connection_router.delete_file_by_id(private_folder_id)
            except Exception:
                logger.warning(
                    "Cleanup: failed to delete private GDrive folder %s",
                    private_folder_id,
                )

        if mock_folder_id is not None:
            try:
                self._connection_router.delete_file_by_id(mock_folder_id)
            except Exception:
                logger.warning(
                    "Cleanup: failed to delete mock GDrive folder %s",
                    mock_folder_id,
                )

        if created_local and dataset_name is not None:
            try:
                self.dataset_manager.delete(dataset_name, require_confirmation=False)
            except Exception:
                logger.warning(
                    "Cleanup: failed to delete local dataset '%s'",
                    dataset_name,
                )

    def _upload_dataset_to_collection(self, dataset, users: list[str] | str) -> str:
        """Upload dataset files to collection folder. Returns the folder ID."""
        from syft_client.sync.connections.drive.gdrive_transport import (
            DatasetCollectionFolder,
        )

        collection_tag = dataset.name

        # Prepare files to upload
        files = {}
        for mock_file in dataset.mock_files:
            if mock_file.exists():
                files[mock_file.name] = mock_file.read_bytes()

        metadata_path = dataset.mock_dir / "dataset.yaml"
        if metadata_path.exists():
            files["dataset.yaml"] = metadata_path.read_bytes()

        if dataset.readme_path and dataset.readme_path.exists():
            files[dataset.readme_path.name] = dataset.readme_path.read_bytes()

        # Compute content hash
        content_hash = DatasetCollectionFolder.compute_hash(files)

        # Create collection folder with hash in name
        folder_id = self._connection_router.owner_create_dataset_collection_folder(
            tag=collection_tag, content_hash=content_hash, owner_email=self.email
        )

        # Upload files
        self._connection_router.owner_upload_dataset_files(
            collection_tag, content_hash, files
        )

        # Share with users
        if users == "any":
            self._connection_router.owner_tag_dataset_collection_as_any(
                collection_tag, content_hash
            )
            self.datasite_owner_syncer._any_shared_datasets.append(
                (collection_tag, content_hash)
            )
            # Share with all already-approved peers
            peer_emails = [p.email for p in self.peer_manager.approved_peers]
            if peer_emails:
                self._connection_router.owner_share_dataset_collection(
                    collection_tag, content_hash, peer_emails
                )
        else:
            self._connection_router.owner_share_dataset_collection(
                collection_tag, content_hash, users
            )

        return folder_id

    def _upload_private_dataset_to_collection(self, dataset) -> str | None:
        """Upload private dataset files to a separate owner-only collection folder.
        Returns the folder ID, or None if no files to upload."""
        from syft_client.sync.connections.drive.gdrive_transport import (
            PrivateDatasetCollectionFolder,
        )

        collection_tag = dataset.name

        # Collect all files in private dir (data, metadata, permissions)
        files = {}
        for f in dataset.private_dir.iterdir():
            if f.is_file():
                files[f.name] = f.read_bytes()

        if not files:
            return None

        content_hash = PrivateDatasetCollectionFolder.compute_hash(files)

        # Create private collection folder (no sharing)
        folder_id = (
            self._connection_router.owner_create_private_dataset_collection_folder(
                tag=collection_tag, content_hash=content_hash, owner_email=self.email
            )
        )

        # Upload files
        self._connection_router.owner_upload_private_dataset_files(
            collection_tag, content_hash, files
        )

        return folder_id

    def delete_dataset(
        self,
        name: str,
        datasite: str | None = None,
        require_confirmation: bool = True,
        sync=True,
    ):
        if self.dataset_manager is None:
            raise ValueError("Dataset manager is not set")
        self.dataset_manager.delete(
            name=name,
            datasite=datasite,
            require_confirmation=require_confirmation,
        )
        # Delete collection folders from Google Drive so DS peers
        # pick up the deletion on their next sync.
        try:
            self._connection_router.owner_delete_dataset_collection(name)
        except Exception:
            logger.warning("Failed to delete dataset collection '%s' from Drive", name)
        try:
            self._connection_router.owner_delete_private_dataset_collection(name)
        except Exception:
            logger.warning(
                "Failed to delete private dataset collection '%s' from Drive",
                name,
            )
        if sync:
            self.sync()

    def share_dataset(self, tag: str, users: list[str] | str, sync=True):
        """
        Share an existing dataset with additional users.

        Args:
            tag: Dataset name
            users: List of email addresses or "any"
            sync: Whether to sync after sharing
        """
        from syft_client.sync.connections.drive.gdrive_transport import (
            DatasetCollectionFolder,
        )

        if self.dataset_manager is None:
            raise ValueError("Dataset manager is not set")

        if not self.has_do_role:
            raise ValueError("Only dataset owners can share datasets")

        # Verify dataset exists
        dataset = self.dataset_manager.get(name=tag, datasite=self.email)
        if dataset is None:
            raise ValueError(f"Dataset {tag} not found")

        # Compute current content hash from local files
        files = {}
        for mock_file in dataset.mock_files:
            if mock_file.exists():
                files[mock_file.name] = mock_file.read_bytes()
        metadata_path = dataset.mock_dir / "dataset.yaml"
        if metadata_path.exists():
            files["dataset.yaml"] = metadata_path.read_bytes()
        if dataset.readme_path and dataset.readme_path.exists():
            files[dataset.readme_path.name] = dataset.readme_path.read_bytes()

        content_hash = DatasetCollectionFolder.compute_hash(files)

        # Share collection
        if users == "any":
            self._connection_router.owner_tag_dataset_collection_as_any(
                tag, content_hash
            )
            self.datasite_owner_syncer._any_shared_datasets.append((tag, content_hash))
            peer_emails = [p.email for p in self.peer_manager.approved_peers]
            if peer_emails:
                self._connection_router.owner_share_dataset_collection(
                    tag, content_hash, peer_emails
                )
        else:
            if isinstance(users, str):
                users = [users]
            self._connection_router.owner_share_dataset_collection(
                tag, content_hash, users
            )

        if sync:
            self.sync()

    def share_private_dataset(self, tag: str, enclave_email: str):
        """Share private dataset files with an enclave via outbox events."""
        if not self.has_do_role:
            raise ValueError("Only data owners can share private datasets")

        with self._sync_file_lock():
            files = self.dataset_manager.get_private_dataset_files(tag)
            events_message = (
                self.datasite_owner_syncer.event_cache.create_events_for_files(files)
            )
            self.datasite_owner_syncer.queue_event_for_syftbox(
                recipients=[enclave_email],
                file_change_events_message=events_message,
            )
            self.datasite_owner_syncer.process_syftbox_events_queue()

    @property
    def datasets(self) -> SyftDatasetManager:
        """
        Get the dataset manager. Automatically calls sync() before returning datasets
        if PRE_SYNC environment variable is set to "true" (case-insensitive).

        PRE_SYNC defaults to "true", so auto-sync is enabled by default.
        To disable auto-sync, set: PRE_SYNC=false
        """
        if self.dataset_manager is None:
            raise ValueError("Dataset manager is not set")

        if os.environ.get("PRE_SYNC", "true").lower() == "true":
            self.sync()

        return self.dataset_manager

    @property
    def _connection_router(self) -> ConnectionRouter:
        # for DOs we have a syncer, for DSs we have a watcher syncer
        if self.datasite_owner_syncer is not None:
            return self.datasite_owner_syncer.connection_router
        else:
            return self.datasite_watcher_syncer.connection_router

    def reset_all_connection_caches(self):
        """Reset GDrive caches on all connection router instances."""
        if self.peer_manager:
            self.peer_manager.connection_router.reset_caches()
        if self.datasite_owner_syncer:
            self.datasite_owner_syncer.connection_router.reset_caches()
        if self.datasite_watcher_syncer:
            self.datasite_watcher_syncer.connection_router.reset_caches()
            if hasattr(self.datasite_watcher_syncer, "datasite_watcher_cache"):
                self.datasite_watcher_syncer.datasite_watcher_cache.connection_router.reset_caches()

    def _clear_caches(self):
        if self.datasite_owner_syncer is not None:
            self.datasite_owner_syncer.event_cache.clear_cache()
        if self.datasite_watcher_syncer is not None:
            self.datasite_watcher_syncer.datasite_watcher_cache.clear_cache()
        self.peer_manager.clear_caches()

    def _broadcast_delete_events(
        self,
        peer_emails: list[str],
        file_hashes: dict,
    ):
        """Broadcast is_deleted=True events for all tracked files to each peer's outbox."""
        from uuid import uuid4
        from syft_client.sync.utils.syftbox_utils import create_event_timestamp

        timestamp = create_event_timestamp()
        events = []
        for path in file_hashes:
            events.append(
                FileChangeEvent(
                    id=uuid4(),
                    path_in_datasite=path,
                    datasite_email=self.email,
                    content=None,
                    old_hash=file_hashes[path],
                    new_hash=None,
                    is_deleted=True,
                    submitted_timestamp=timestamp,
                    timestamp=timestamp,
                )
            )

        if not events:
            return

        msg = FileChangeEventsMessage(events=events)
        for peer_email in peer_emails:
            try:
                self._connection_router.owner_write_event_messages_to_outbox(
                    peer_email, msg
                )
            except Exception:
                pass

    def delete_syftbox(
        self, verbose: bool = True, broadcast_delete_events: bool = True
    ):
        """
        Delete all SyftBox state: Google Drive files, local caches, and local folder.

        Due to Google Drive's eventual consistency, files can become orphaned when
        their parent folder is deleted before they're fully registered. We use two
        strategies to ensure complete cleanup:
        1. Gather all files by traversing the SyftBox folder hierarchy
        2. Find files by name pattern (catches orphaned files from any location)

        Args:
            verbose: Print deletion progress.
            broadcast_delete_events: If True (default), broadcast is_deleted events
                to all approved peers before deleting. Set False for test cleanup.
        """
        # Capture state before deletion (needed for broadcast)
        peer_emails = []
        file_hashes = {}
        if broadcast_delete_events and self.has_do_role:
            peer_emails = [p.email for p in self.peer_manager.approved_peers]
            file_hashes = dict(self.datasite_owner_syncer.event_cache.file_hashes)

        # Get files by folder hierarchy
        folder_file_ids = set(self._connection_router.gather_all_file_and_folder_ids())

        # Also find syft files by name pattern (catches orphaned files)
        orphaned_file_ids = set(self._connection_router.find_orphaned_message_files())

        # Combine both sets and delete from Google Drive
        all_file_ids = list(folder_file_ids | orphaned_file_ids)

        start = time.time()
        self._connection_router.delete_multiple_files_by_ids(all_file_ids)
        end = time.time()
        if verbose:
            orphan_count = len(orphaned_file_ids - folder_file_ids)
            print(
                f"Deleted {len(all_file_ids)} files/folders in {end - start:.2f}s",
                end="",
            )
            if orphan_count > 0:
                print(f" (including {orphan_count} orphaned)")
            else:
                print()

        # Broadcast delete events after file deletion but before cache reset
        if broadcast_delete_events and self.has_do_role and peer_emails and file_hashes:
            self._broadcast_delete_events(peer_emails, file_hashes)

        # Clear in-memory caches and filesystem cache contents
        self._clear_caches()
        self.reset_all_connection_caches()

        # Delete local syftbox folder and cache directories
        self._delete_local_dirs()

        print(
            f"{_ANSI_RED}Done. If you are also running syft-bg, make sure to call "
            f"syft_bg.reset() before delete_syftbox().{_ANSI_RESET}"
        )

    def _delete_local_dirs(self):
        """Delete local syftbox folder and cache directories."""
        from syft_client.sync.peers.peer_store import CRYPTO_KEYS_PATH

        syftbox_name = self.syftbox_folder.name
        syftbox_parent = self.syftbox_folder.parent

        dirs_to_delete = [
            self.syftbox_folder,  # main syftbox folder (datasets, private, etc.)
            syftbox_parent / f"{syftbox_name}-events",  # DO event cache
            syftbox_parent / f"{syftbox_name}-event-messages",  # DS event cache
        ]
        for d in dirs_to_delete:
            if d.exists():
                shutil.rmtree(d)

        # Persistent encryption keys, so a fresh state regenerates a new identity.
        CRYPTO_KEYS_PATH.unlink(missing_ok=True)

    # =========================================================================
    # CHECKPOINT METHODS
    # =========================================================================

    def create_checkpoint(self):
        """
        Create a checkpoint of the current state and upload to Google Drive.

        A checkpoint is a snapshot of all files and their hashes. When logging in,
        the client will download the checkpoint instead of all historical events,
        significantly speeding up the initial sync.

        Only available for Data Owners (DO).

        Returns:
            The created Checkpoint object.

        Raises:
            ValueError: If called on a Data Scientist client.
        """
        if not self.has_do_role:
            raise ValueError("Checkpoints can only be created by Data Owners")
        with self._sync_file_lock():
            return self.datasite_owner_syncer.create_checkpoint()

    def compact_outboxes_if_needed(
        self, min_messages: int = MIN_MESSAGES_COMPACT
    ) -> dict[str, int]:
        """Merge accumulated event messages in each approved peer's outbox.

        This compacts each peer's outbox into a
        single message when it has at least `min_messages` files.

        Returns a mapping of recipient_email -> number of source messages
        compacted (0 if that peer was below threshold or skipped).
        Only available for Data Owners.
        """
        if not self.has_do_role:
            return {}
        with self._sync_file_lock():
            return {
                peer.email: self.datasite_owner_syncer.compact_outbox_if_needed(
                    peer.email, min_messages=min_messages
                )
                for peer in self.peer_manager.approved_peers
            }

    def should_create_checkpoint(self, threshold: int = 50) -> bool:
        """
        Check if a checkpoint should be created based on event count.

        Args:
            threshold: Create checkpoint if events since last checkpoint >= threshold.

        Returns:
            True if checkpoint should be created.
        """
        if not self.has_do_role:
            return False
        return self.datasite_owner_syncer.should_create_checkpoint(threshold)

    def try_create_checkpoint(self, threshold: int = 50):
        """
        Try to create a checkpoint if the event count exceeds the threshold.

        This is useful for automatic checkpoint creation after syncs.

        Args:
            threshold: Create checkpoint if events since last checkpoint >= threshold.

        Returns:
            The created Checkpoint, or None if not needed or not a DO.
        """
        if not self.has_do_role:
            return None
        return self.datasite_owner_syncer.try_create_checkpoint(threshold)

    def _get_all_peer_platforms(self) -> List[BasePlatform]:
        all_platforms = set(
            [plat for p in self.peer_manager.approved_peers for plat in p.platforms]
        )
        return list(all_platforms)

    def _resolve_path(self, path: str | Path) -> Path:
        return resolve_path(path, syftbox_folder=self.syftbox_folder)

    def _resolve_dataset_owners_for_name(self, dataset_name: str) -> str | None:
        matches = []
        for dataset in self.dataset_manager.get_all():
            if dataset.name == dataset_name:
                matches.append(dataset.owner)
        return matches

    def _copy(self):
        from copy import deepcopy

        new_config = deepcopy(self.config)
        new_manager = SyftboxManager.from_config(new_config)
        if not isinstance(self._connection_router.connections[0], GDriveConnection):
            raise ValueError("Only GDriveConnections can be copied")
        if isinstance(
            self._connection_router.connections[0].drive_service,
            mock_drive_service.MockDriveService,
        ):
            # Create new connection pointing to the same backing store
            drive_service = self._connection_router.connections[0].drive_service
            new_do_connection = GDriveConnection.from_service(self.email, drive_service)
            new_manager._add_connection(new_do_connection)
        return new_manager

    # def resolve_dataset_path(
    #     self, dataset_name: str, owner_email: str | None = None
    # ) -> Path:
    #     if owner_email is None:
    #         owner_emails = self._resolve_dataset_owners_for_name(dataset_name)
    #         if len(owner_emails) == 1:
    #             owner_email = owner_emails[0]
    #         else:
    #             raise ValueError(
    #                 f"Dataset {dataset_name} has 0 or multiple owners: {owner_emails}, please specify the owner_email"
    #             )

    #     return resolve_dataset_path(
    #         dataset_name, syftbox_folder=self.syftbox_folder, owner_email=owner_email
    #     )
