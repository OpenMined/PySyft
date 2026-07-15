from pathlib import Path
import fcntl
from syft_client.sync.peers.peer_store import PeerStore
from syft_client.sync.utils.path_filters import is_normal_syncable_path
from syft_client.sync.callback_mixin import BaseModelCallbackMixin
import logging
import shutil
from contextlib import contextmanager
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.utils import resolve_path
from concurrent.futures import ThreadPoolExecutor
import time
from pydantic import ConfigDict
from syft_client.sync.platforms.base_platform import BasePlatform
from pydantic import BaseModel, PrivateAttr
from typing import List, Optional, cast
from syft_client.sync.sync.collection_spec import CollectionSyncSpec
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
    PeerManager,
    PeerManagerConfig,
)
from syft_client.sync.version.version_info import VersionInfo
from syft_client.version import VERSION_FILE_NAME
import os

logger = logging.getLogger(__name__)

COLAB_DEFAULT_SYFTBOX_FOLDER = Path("/")
JUPYTER_DEFAULT_SYFTBOX_FOLDER = Path.home() / "SyftBox"

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

    @classmethod
    def for_colab(
        cls,
        email: str,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        encryption: bool = False,
        crypto_keys_path: Path | None = None,
        skip_peer_on_patch_version_diff: Optional[
            bool
        ] = None,  # None: value is determined by the role
        force_ignore_peer_version: bool = False,
        collection_specs: list["CollectionSyncSpec"] | None = None,
    ):
        if not has_ds_role and not has_do_role:
            raise ValueError("At least one of has_ds_role or has_do_role must be True")

        # No collection specs by default: the generic sync engine knows nothing
        # about datasets. syft-rds registers its dataset spec at initialization
        # (see syft_rds.config.DATASET_COLLECTION_SPECS).
        if collection_specs is None:
            collection_specs = []

        syftbox_folder = get_colab_default_syftbox_folder(email)
        use_in_memory_cache = False
        collections_folder = (
            syftbox_folder / email / collection_specs[0].local_subpath
            if collection_specs
            else None
        )
        connection_configs = [GdriveConnectionConfig(email=email, token_path=None)]
        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            collection_specs=collection_specs,
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
                collection_specs=collection_specs,
                connection_configs=connection_configs,
            ),
        )
        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
            connection_configs=connection_configs,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            use_encryption=encryption,
            crypto_keys_path=crypto_keys_path,
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
        crypto_keys_path: Path | None = None,
        skip_peer_on_patch_version_diff: Optional[
            bool
        ] = None,  # None: value is determined by the role
        force_ignore_peer_version: bool = False,
        collection_specs: list["CollectionSyncSpec"] | None = None,
    ):
        if not has_ds_role and not has_do_role:
            raise ValueError("At least one of has_ds_role or has_do_role must be True")

        # No collection specs by default: the generic sync engine knows nothing
        # about datasets. syft-rds registers its dataset spec at initialization
        # (see syft_rds.config.DATASET_COLLECTION_SPECS).
        if collection_specs is None:
            collection_specs = []

        syftbox_folder = get_jupyter_default_syftbox_folder(email)
        collections_folder = (
            syftbox_folder / email / collection_specs[0].local_subpath
            if collection_specs
            else None
        )

        connection_configs = [
            GdriveConnectionConfig(email=email, token_path=token_path)
        ]
        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            collection_specs=collection_specs,
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
                collection_specs=collection_specs,
                connection_configs=connection_configs,
            ),
        )
        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
            connection_configs=connection_configs,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            use_encryption=encryption,
            crypto_keys_path=crypto_keys_path,
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
        collection_specs: list["CollectionSyncSpec"] | None = None,
    ):
        # No collection specs by default: the generic sync engine knows nothing
        # about datasets. syft-rds registers its dataset spec at initialization
        # (see syft_rds.config.DATASET_COLLECTION_SPECS).
        if collection_specs is None:
            collection_specs = []

        syftbox_folder = syftbox_folder or random_syftbox_folder_for_testing()
        email = email or random_email()
        collections_folder = (
            syftbox_folder / email / collection_specs[0].local_subpath
            if collection_specs
            else None
        )

        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            collection_specs=collection_specs,
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
                collection_specs=collection_specs,
            ),
        )

        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
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
        collection_specs: list["CollectionSyncSpec"] | None = None,
    ):
        # No collection specs by default: the generic sync engine knows nothing
        # about datasets. syft-rds registers its dataset spec at initialization
        # (see syft_rds.config.DATASET_COLLECTION_SPECS).
        if collection_specs is None:
            collection_specs = []

        syftbox_folder = syftbox_folder or random_syftbox_folder_for_testing()
        email = email or random_email()
        collections_folder = (
            Path(syftbox_folder) / email / collection_specs[0].local_subpath
            if collection_specs
            else None
        )
        connection_configs = [
            GdriveConnectionConfig(email=email, token_path=token_path)
        ]
        datasite_owner_syncer_config = DatasiteOwnerSyncerConfig(
            email=email,
            syftbox_folder=syftbox_folder,
            collections_folder=collections_folder,
            collection_specs=collection_specs,
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
                collection_specs=collection_specs,
                connection_configs=connection_configs,
            ),
        )

        peer_manager_config = PeerManagerConfig(
            syftbox_folder=syftbox_folder,
            email=email,
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
            peer_manager_config=peer_manager_config,
        )


class SyftboxManager(BaseModelCallbackMixin):
    # needed for peers
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_writer: FileWriter
    syftbox_folder: Path
    email: str
    dev_mode: bool = False
    datasite_watcher_syncer: DatasiteWatcherSyncer | None = None

    datasite_owner_syncer: DatasiteOwnerSyncer | None = None
    job_file_change_handler: JobFileChangeHandler | None = None
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
        "peer_manager",
        "peers",
        "add_peer",
        "load_peers",
        "approve_peer_request",
        "reject_peer_request",
        "sync",
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

        if config.has_do_role:
            datasite_owner_syncer = DatasiteOwnerSyncer.from_config(
                config.datasite_owner_syncer_config
            )

            job_file_change_handler = JobFileChangeHandler()

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
        crypto_keys_path: Path | None = None,
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
                crypto_keys_path=crypto_keys_path,
                skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
                force_ignore_peer_version=force_ignore_peer_version,
            )
        )
        return manager

    @classmethod
    def for_jupyter(
        cls,
        email: str,
        has_ds_role: bool = False,
        has_do_role: bool = False,
        token_path: Path | None = None,
        encryption: bool = False,
        crypto_keys_path: Path | None = None,
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
                crypto_keys_path=crypto_keys_path,
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
        collection_specs: list["CollectionSyncSpec"] | None = None,
    ):
        receiver_config = SyftboxManagerConfig.for_google_drive_testing_connection(
            email=do_email,
            syftbox_folder=base_path1,
            use_in_memory_cache=use_in_memory_cache,
            token_path=do_token_path,
            has_ds_role=False,
            has_do_role=True,
            check_versions=check_versions,
            collection_specs=collection_specs,
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
            collection_specs=collection_specs,
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
        collection_specs: list["CollectionSyncSpec"] | None = None,
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
            collection_specs=collection_specs,
        )

        ds_config = SyftboxManagerConfig._base_config_for_testing(
            email=email2,
            syftbox_folder=base_path2,
            has_ds_role=True,
            has_do_role=False,
            use_in_memory_cache=use_in_memory_cache,
            check_versions=check_versions,
            collection_specs=collection_specs,
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
        self._emit_peers_loaded()
        if self.has_do_role:
            self._post_approve_peer_do(peer_email)
        if sync:
            self.sync()

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
        self._emit_peers_loaded()

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
        self._emit_peers_loaded()
        self._post_approve_peer_do(email_or_peer)

    def _post_approve_peer_do(self, email_or_peer: str | Peer):
        peer_email = (
            email_or_peer if isinstance(email_or_peer, str) else email_or_peer.email
        )
        self._emit("peer_approved", peer_email)

    def _emit_peers_loaded(self) -> None:
        self._emit("peers_loaded")

    def reject_peer_request(self, email_or_peer: str | Peer):
        """Reject a pending peer request. Delegates to PeerManager."""
        self.peer_manager.reject_peer_request(email_or_peer)

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
        # Encryption keys live in <syftbox_folder>/<email>/private/crypto_keys.json,
        # so deleting the syftbox folder above already removes them — a fresh state
        # regenerates a new identity.

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
