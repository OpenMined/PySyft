"""The Remote Data Science client."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from syft.sync.syftbox_manager import SyftboxManager
from syft_datasets.dataset_manager import (
    DATASET_COLLECTION_PREFIX,
    PRIVATE_DATASET_COLLECTION_PREFIX,
)
from syft.sync.utils.pre_submit_scan import run_pre_submit_check
from syft.sync.version.peer_manager import CompatAction
from syft_job.client import JobClient
from syft_job.job_runner import SyftJobRunner
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_datasets.dataset_ref import DatasetNotFoundError
from syft_rds.config import (
    DATASET_COLLECTION_SPECS,
    MOCK_DATASET_SPEC,
    PRIVATE_DATASET_SPEC,
    SyftRDSClientConfig,
    dataset_variant,
)

logger = logging.getLogger(__name__)


class SyftRDSClient(BaseModel):
    # Holds live service objects (sync engine + RDS-owned managers), not
    # serializable data, so arbitrary types are allowed.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The nested generic sync engine (composition).
    sync_engine: SyftboxManager
    # RDS-owned domain managers, exposed so consumers (e.g. syft-enclave, which
    # wraps the job client) can read/replace them.
    job_client: JobClient
    job_runner: SyftJobRunner | None = None
    dataset_manager: SyftDatasetManager

    def model_post_init(self, __context: Any) -> None:
        # React to sync-engine peer lifecycle events with RDS-owned logic.
        self.sync_engine.on("peer_approved", self._on_peer_approved)
        self.sync_engine.on("peers_loaded", self._on_peers_loaded)
        # React to collection restores so RDS can run dataset-specific post-processing
        # (the private dataset's data_dir fixup) without the sync core knowing datasets.
        if self.sync_engine.datasite_owner_syncer is not None:
            self.sync_engine.datasite_owner_syncer.on(
                "collection_restored", self._on_collection_restored
            )

    @classmethod
    def from_config(cls, config: "SyftRDSClientConfig") -> "SyftRDSClient":
        sync_engine = SyftboxManager.from_config(config.sync)
        # The job client selects a job protocol version per peer from the live
        # peer-schema map, as the dataset manager does for dataset layouts.
        job_client = JobClient.from_config(
            config.job,
            peer_schemas=sync_engine.peer_manager.live_peer_schemas("syft-job"),
        )
        job_runner = (
            SyftJobRunner.from_config(config.job) if config.sync.has_do_role else None
        )
        # The dataset manager gets the live peer-schema map, as the job client
        # does. It selects a layout for each peer, and the transport carries one
        # collection for each layout.
        dataset_manager = SyftDatasetManager.from_config(
            config.dataset,
            peer_schemas=sync_engine.peer_manager.live_peer_schemas("syft-dataset"),
        )
        return cls(
            sync_engine=sync_engine,
            job_client=job_client,
            job_runner=job_runner,
            dataset_manager=dataset_manager,
        )

    @classmethod
    def _build_rds_pair_from_managers(
        cls, ds_mgr: SyftboxManager, do_mgr: SyftboxManager
    ) -> tuple["SyftRDSClient", "SyftRDSClient"]:
        """Wrap a paired ``(ds, do)`` ``SyftboxManager`` tuple into RDS clients.

        Peers were already approved/loaded during pairing (before our callbacks
        were registered), so we replay the DO-side setup so its owned
        ``JobClient`` has the DS job folders + install sources.
        """

        def _build(mgr: SyftboxManager) -> "SyftRDSClient":
            assert mgr.config is not None, "paired managers are built from a config"
            # Compose via the config so the sub-configs are derived (and
            # alignment-checked) exactly as they are on the from_config path.
            config = SyftRDSClientConfig._compose(mgr.config)
            return cls(
                sync_engine=mgr,
                job_client=JobClient.from_config(
                    config.job,
                    peer_schemas=mgr.peer_manager.live_peer_schemas("syft-job"),
                ),
                job_runner=(
                    SyftJobRunner.from_config(config.job) if mgr.has_do_role else None
                ),
                dataset_manager=SyftDatasetManager.from_config(
                    config.dataset,
                    peer_schemas=mgr.peer_manager.live_peer_schemas("syft-dataset"),
                ),
            )

        ds_rds = _build(ds_mgr)
        do_rds = _build(do_mgr)
        do_rds._on_peers_loaded()
        for peer in do_mgr.peer_manager.approved_peers:
            do_rds._on_peer_approved(peer.email)
        ds_rds._on_peers_loaded()
        return ds_rds, do_rds

    @classmethod
    def pair_with_mock_drive_service_connection(
        cls, **kwargs: Any
    ) -> tuple["SyftRDSClient", "SyftRDSClient"]:
        """(ds, do) pair of self-contained RDS clients sharing one mock Drive."""
        ds_mgr, do_mgr = SyftboxManager.pair_with_mock_drive_service_connection(
            collection_specs=DATASET_COLLECTION_SPECS, **kwargs
        )
        return cls._build_rds_pair_from_managers(ds_mgr, do_mgr)

    @classmethod
    def _pair_with_google_drive_testing_connection(
        cls, **kwargs: Any
    ) -> tuple["SyftRDSClient", "SyftRDSClient"]:
        """(ds, do) pair of self-contained RDS clients sharing a REAL Google Drive."""
        ds_mgr, do_mgr = SyftboxManager._pair_with_google_drive_testing_connection(
            collection_specs=DATASET_COLLECTION_SPECS, **kwargs
        )
        return cls._build_rds_pair_from_managers(ds_mgr, do_mgr)

    def __dir__(self) -> list[str]:
        """Public API only, without pydantic's machinery."""
        names = set(type(self).model_fields)
        for klass in type(self).__mro__:
            if klass is BaseModel:
                break
            names.update(
                name
                for name, value in vars(klass).items()
                if not name.startswith(("_", "model_"))
                and not isinstance(value, (classmethod, staticmethod))
            )
        return sorted(names)

    # ------------------------------------------------------------------ #
    # delegated identity + sync surface (owned by the generic core)
    # ------------------------------------------------------------------ #
    @property
    def email(self) -> str:
        return self.sync_engine.email

    @property
    def syftbox_folder(self) -> Path:
        return self.sync_engine.syftbox_folder

    @property
    def has_do_role(self) -> bool:
        return self.sync_engine.has_do_role

    @property
    def has_ds_role(self) -> bool:
        return self.sync_engine.has_ds_role

    @property
    def peer_manager(self) -> Any:
        return self.sync_engine.peer_manager

    @property
    def peers(self) -> Any:
        """Combined peer list (approved + requests). Delegates to the sync engine.

        Auto-syncs first when ``PRE_SYNC`` is enabled (the engine's behavior).
        """
        return self.sync_engine.peers

    def sync(self, *args: Any, **kwargs: Any) -> Any:
        return self.sync_engine.sync(*args, **kwargs)

    def add_peer(self, *args: Any, **kwargs: Any) -> Any:
        return self.sync_engine.add_peer(*args, **kwargs)

    def approve_peer_request(self, *args: Any, **kwargs: Any) -> Any:
        return self.sync_engine.approve_peer_request(*args, **kwargs)

    def reject_peer_request(self, *args: Any, **kwargs: Any) -> Any:
        return self.sync_engine.reject_peer_request(*args, **kwargs)

    def load_peers(self, *args: Any, **kwargs: Any) -> Any:
        return self.sync_engine.load_peers(*args, **kwargs)

    def delete_syftbox(self, *args: Any, **kwargs: Any) -> Any:
        return self.sync_engine.delete_syftbox(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # peer lifecycle callbacks (RDS-owned reactions to sync events)
    # ------------------------------------------------------------------ #
    def _on_peer_approved(self, peer_email: str) -> None:
        """A peer was approved: set up their DS job folder and share datasets."""
        if self.has_do_role:
            self.job_client.setup_ds_job_folder_as_do(peer_email)
            self._share_any_datasets_with_peer(peer_email)

    def _on_collection_restored(self, prefix: str, tag: str, local_dir: Any) -> None:
        """A collection was restored from the backend by the owner-syncer.

        For the PRIVATE dataset collection, rewrite ``private_metadata.yaml``'s
        machine-specific ``data_dir`` to the current local path, so job execution
        can locate the real data after a restore onto a new machine/path. This is
        dataset domain knowledge, so it lives here (rds) rather than the sync core.
        """
        if prefix != PRIVATE_DATASET_COLLECTION_PREFIX:
            return

        metadata_path = Path(local_dir) / "private_metadata.yaml"
        if not metadata_path.exists():
            return
        data = yaml.safe_load(metadata_path.read_text())
        if not data:
            # Missing or empty file: nothing to rewrite (and guards against
            # yaml.safe_load returning None on an empty file).
            return
        expected_dir = str(local_dir)
        if data.get("data_dir") != expected_dir:
            data["data_dir"] = expected_dir
            metadata_path.write_text(yaml.safe_dump(data, indent=2, sort_keys=False))

    def _on_peers_loaded(self, *args: Any, **kwargs: Any) -> None:
        """Peers were (re)loaded: copy each peer's advertised install source
        into our owned job client so submitted run.sh references the DO's path."""
        for peer in self.sync_engine.peer_manager.syncable_peers:
            if peer.version:
                self.job_client.peer_install_sources[peer.email] = (
                    peer.version.syft_client_install_source
                )

    def _share_any_datasets_with_peer(self, peer_email: str) -> None:
        """Share all datasets tagged 'any' with a specific peer.

        Google Drive "anyone with link" files are not discoverable via search,
        so explicit user sharing is added. Reads the cache populated during
        ``pull_initial_state()`` in the nested DatasiteOwnerSyncer.
        """
        for (
            wire_prefix,
            tag,
            content_hash,
        ) in self.sync_engine.datasite_owner_syncer.any_shared_collections:
            try:
                self.sync_engine.share_collection(
                    wire_prefix, tag, content_hash, [peer_email]
                )
            except Exception:
                # One collection failing (missing folder, quota, network) must
                # not stop us sharing the rest with this peer. "alreadyShared"
                # is already handled in _batch_add_permissions, so anything
                # reaching here is a real failure worth a traceback.
                logger.exception(
                    "Failed to share collection %r with %s", tag, peer_email
                )

    # ------------------------------------------------------------------ #
    # job product surface (RDS-owned)
    # ------------------------------------------------------------------ #
    def submit_bash_job(
        self,
        user: str,
        script: str,
        job_name: str = "",
        force_submission: bool = False,
        ignore_peer_version: bool = False,
    ):
        # Check version compatibility before submission (uses cached versions)
        if not force_submission:
            result = self.sync_engine.peer_manager.get_peer_compatibility_status(
                user,
                action=CompatAction.SUBMIT,
                ignore_peer_version=ignore_peer_version,
            )
            result.raise_on_skip(operation="submit job")
            result.maybe_warn()
        job_dir = self.job_client.submit_bash_job(user, script, job_name=job_name)
        self.sync_engine.push_job_files(job_dir)

    def submit_python_job(
        self,
        user: str,
        code_path: str,
        job_name: str | None = "",
        dependencies: list[str] | None = None,
        entrypoint: str | None = None,
        force_submission: bool = False,
        ignore_peer_version: bool = False,
    ):
        peer_emails = {p.email for p in self.sync_engine.peer_manager.syncable_peers}
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
            result = self.sync_engine.peer_manager.get_peer_compatibility_status(
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
        self.sync_engine.push_job_files(job_dir)

        print("\n✅ Job submitted successfully!")
        print("   Status : inbox (waiting for DO to review)")
        print(f"\n⏳ Next step: wait for {user} to approve and run it.")
        print("   Check progress with: client.jobs")

    @property
    def _pre_sync_enabled(self) -> bool:
        """Whether accessors auto-sync (disabled by setting ``PRE_SYNC=false``)."""
        return os.environ.get("PRE_SYNC", "true").lower() == "true"

    @property
    def jobs(self) -> Any:
        """List of jobs. Auto-syncs first unless PRE_SYNC=false."""
        if self._pre_sync_enabled:
            self.sync_engine.sync()
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
        """Process approved jobs (DO only). Auto-syncs after unless PRE_SYNC=false."""
        if not self.has_do_role:
            raise ValueError("Only dataset owners can process approved jobs")
        if self.job_runner is None:
            raise ValueError("Job runner is not configured for this client")

        skip_job_names = []

        if not force_execution:
            approved_jobs = [
                job for job in self.job_client.jobs if job.status == "approved"
            ]
            for job in approved_jobs:
                result = self.sync_engine.peer_manager.get_peer_compatibility_status(
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

        if self._pre_sync_enabled:
            self.sync_engine.sync()

    # ------------------------------------------------------------------ #
    # dataset product surface (RDS-owned)
    # ------------------------------------------------------------------ #
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
        mock_folder_ids: list[str] = []
        private_folder_ids: list[str] = []

        try:
            # Create the dataset locally, in one layout for each protocol
            # version the audience reads.
            created = self.dataset_manager.create_all(
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
            # The newest copy is the one to hand back to the owner.
            dataset = created[max(created, key=int)]
            dataset_name = dataset.name

            # Each copy gets its own collection. The private data of a copy goes
            # up with it, because the metadata of that copy points at it.
            for protocol_version in sorted(created, key=int):
                copy = created[protocol_version]
                mock_folder_ids.append(self._upload_dataset_to_collection(copy, users))
                if upload_private:
                    private_folder_id = self._upload_private_dataset_to_collection(copy)
                    if private_folder_id is not None:
                        private_folder_ids.append(private_folder_id)

            if sync:
                self.sync()

            return dataset

        except Exception:
            logger.error(
                "Failed to create dataset%s, cleaning up",
                f" '{dataset_name}'" if dataset_name else "",
            )
            self._cleanup_failed_dataset_creation(
                dataset_name, created_local, mock_folder_ids, private_folder_ids
            )
            raise

    def _cleanup_failed_dataset_creation(
        self,
        dataset_name: str | None,
        created_local: bool,
        mock_folder_ids: list[str],
        private_folder_ids: list[str],
    ) -> None:
        """Best-effort cleanup after a failed create_dataset, in reverse order."""
        for private_folder_id in reversed(private_folder_ids):
            try:
                self.sync_engine.delete_file_by_id(private_folder_id)
            except Exception:
                logger.warning(
                    "Cleanup: failed to delete private GDrive folder %s",
                    private_folder_id,
                )

        for mock_folder_id in reversed(mock_folder_ids):
            try:
                self.sync_engine.delete_file_by_id(mock_folder_id)
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

    def _create_and_upload_collection(
        self, prefix: str, tag: str, files: dict[str, bytes]
    ) -> tuple[str, str]:
        """Create a hash-named collection folder and upload its files.

        Returns ``(folder_id, content_hash)``.
        """
        from syft.sync.connections.drive.gdrive_transport import (
            CollectionFolder,
        )

        content_hash = CollectionFolder.compute_hash(files)
        folder_id = self.sync_engine.create_collection_folder(
            prefix, tag=tag, content_hash=content_hash
        )
        self.sync_engine.upload_collection_files(prefix, tag, content_hash, files)
        return folder_id, content_hash

    def _collect_mock_files(self, dataset) -> dict[str, bytes]:
        """Read a dataset's mock files, metadata and readme into a name->bytes map."""
        files = {}
        for mock_file in dataset.mock_files:
            if mock_file.exists():
                files[mock_file.name] = mock_file.read_bytes()

        metadata_path = dataset.mock_dir / "dataset.yaml"
        if metadata_path.exists():
            files["dataset.yaml"] = metadata_path.read_bytes()

        if dataset.readme_path and dataset.readme_path.exists():
            files[dataset.readme_path.name] = dataset.readme_path.read_bytes()
        return files

    def _share_dataset_collection(
        self, wire_prefix: str, tag: str, content_hash: str, users: list[str] | str
    ) -> None:
        """Share one layout of a dataset with ``users``, or tag it ``"any"`` and
        share with all already-approved peers.

        Every layout is shared with the whole audience, so a peer that upgrades
        later moves to the newer layout with no action by the owner.
        """
        if users == "any":
            self.sync_engine.tag_collection_as_any(wire_prefix, tag, content_hash)
            self.sync_engine.datasite_owner_syncer.register_any_shared_collection(
                wire_prefix, tag, content_hash
            )
            peer_emails = [
                p.email for p in self.sync_engine.peer_manager.approved_peers
            ]
            if peer_emails:
                self.sync_engine.share_collection(
                    wire_prefix, tag, content_hash, peer_emails
                )
        else:
            if isinstance(users, str):
                users = [users]
            self.sync_engine.share_collection(wire_prefix, tag, content_hash, users)

    def _upload_dataset_to_collection(self, dataset, users: list[str] | str) -> str:
        """Upload one protocol copy of a dataset. Returns the folder ID.

        Each copy gets its own collection, named for its layout, so a peer picks
        the newest copy it can read.
        """
        variant = dataset_variant(dataset.protocol_version)
        wire_prefix = MOCK_DATASET_SPEC.wire_prefix(variant)
        files = self._collect_mock_files(dataset)
        folder_id, content_hash = self._create_and_upload_collection(
            wire_prefix, dataset.name, files
        )
        self._share_dataset_collection(wire_prefix, dataset.name, content_hash, users)
        return folder_id

    def _upload_private_dataset_to_collection(self, dataset) -> str | None:
        """Upload the private files of one protocol copy to an owner-only collection.

        The copies hold separate private directories, so one upload of the newest
        would leave the others local only and a cold start would not restore them.
        Returns the folder ID, or None if there are no files to upload.
        """
        collection_tag = dataset.name

        # Collect all files in private dir (data, metadata, permissions)
        files = {}
        for f in dataset.private_dir.iterdir():
            if f.is_file():
                files[f.name] = f.read_bytes()

        if not files:
            return None

        # Private collection: no sharing step.
        variant = dataset_variant(dataset.protocol_version)
        folder_id, _ = self._create_and_upload_collection(
            PRIVATE_DATASET_SPEC.wire_prefix(variant), collection_tag, files
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
            self.sync_engine.delete_collection(DATASET_COLLECTION_PREFIX, name)
        except Exception:
            logger.warning("Failed to delete dataset collection '%s' from Drive", name)
        try:
            self.sync_engine.delete_collection(PRIVATE_DATASET_COLLECTION_PREFIX, name)
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

        if self.dataset_manager is None:
            raise ValueError("Dataset manager is not set")

        if not self.has_do_role:
            raise ValueError("Only dataset owners can share datasets")

        # Verify dataset exists
        dataset = self.dataset_manager.get(name=tag, datasite=self.email)
        if dataset is None:
            raise ValueError(f"Dataset {tag} not found")

        if users != "any" and isinstance(users, str):
            users = [users]

        # A dataset has one collection for each layout it was written in. Share
        # them all, so a peer of any supported version finds a copy. The listing
        # gives the hash of each copy, so no hash is recomputed here.
        collections = self._mock_collections_for(tag)
        if not collections:
            raise ValueError(f"No uploaded collection found for dataset {tag}")

        # A share is a change of audience. The layouts were decided by the
        # audience at create time, so a new peer whose protocol reads none of
        # them would get a grant on a folder its client never even lists.
        # Materialize what is missing first, then share everything.
        if self._ensure_dataset_layouts_for(
            tag, users, {self._protocol_of(c) for c in collections}
        ):
            collections = self._mock_collections_for(tag)

        for collection in collections:
            self._share_dataset_collection(
                MOCK_DATASET_SPEC.wire_prefix(collection.variant),
                tag,
                collection.content_hash,
                users,
            )

        if sync:
            self.sync()

    def _mock_collections_for(self, tag: str) -> list:
        """Every uploaded mock-data layout of one dataset."""
        return [
            c
            for c in self.sync_engine._connection_router.owner_list_all_collections_with_permissions(
                DATASET_COLLECTION_PREFIX
            )
            if c.tag == tag
        ]

    def _private_collections_for(self, tag: str) -> list:
        """Every uploaded private-data layout of one dataset."""
        return [
            c
            for c in self.sync_engine._connection_router.owner_list_all_collections_with_permissions(
                PRIVATE_DATASET_COLLECTION_PREFIX
            )
            if c.tag == tag
        ]

    @staticmethod
    def _protocol_of(collection) -> str:
        """The dataset protocol version a collection's wire variant stands for."""
        return collection.variant.removeprefix("v") or "0"

    def _ensure_dataset_layouts_for(
        self, tag: str, users: list[str] | str, existing_versions: set[str]
    ) -> bool:
        """Materialize any layout the audience reads but no existing copy serves.

        A peer reads every layout at or below its negotiated protocol version,
        so a copy is only missing when no uploaded collection sits at or below
        the version a peer reads. The new copy uploads unshared; the caller
        shares every collection uniformly afterwards. Returns whether a copy
        was added.
        """
        storage = self.dataset_manager.storage
        peer_emails = self.dataset_manager._peer_emails(users)
        needed = storage.target_protocol_versions_for_peers(peer_emails)
        missing = {
            version
            for version in needed
            if not any(int(e) <= int(version) for e in existing_versions)
        }
        if not missing:
            return False

        for protocol_version in sorted(missing, key=int):
            self._materialize_dataset_copy(tag, protocol_version, users)
        return True

    def _materialize_dataset_copy(
        self, tag: str, protocol_version: str, users: list[str] | str
    ) -> None:
        """Create and upload one layout copy of an existing dataset.

        The copy uploads unshared; sharing stays with the caller. Each copy
        holds its own private directory, so the copy gets its private
        collection iff the dataset's copies are drive-backed -- then a cold
        start restores it like any other.

        The layout may already be on disk with no collection of its own: an
        upload can fail after the migrate, and `migrate` is public. A second
        write of the same layout raises, so an existing copy is read and
        uploaded instead. Permissions are re-applied either way, because a
        migrate re-applies them and both paths must leave the same state.
        """
        storage = self.dataset_manager.storage
        try:
            ref = storage.find_dataset_ref(
                self.email, tag, protocol_version=protocol_version
            )
        except DatasetNotFoundError:
            copy = self.dataset_manager.migrate(tag, protocol_version, users=users)
        else:
            copy = storage.read_dataset(ref)
            self.dataset_manager._set_new_dataset_permissions(dataset=copy, users=users)
        self._upload_dataset_to_collection(copy, users=[])
        if self._private_collections_for(tag):
            self._upload_private_dataset_to_collection(copy)

    def share_private_dataset(self, tag: str, enclave_email: str):
        """Share private dataset files with an enclave via outbox events.

        The files ship at the layout the enclave reads: the newest local copy
        at or below its negotiated dataset protocol, materialized first when
        no copy qualifies. An enclave without a known schema is assumed to run
        the current protocol, the same policy as jobs.
        """
        if not self.has_do_role:
            raise ValueError("Only data owners can share private datasets")

        with self.sync_engine.sync_file_lock():
            protocol_version = self._private_share_protocol_version(tag, enclave_email)
            files = self.dataset_manager.get_private_dataset_files(
                tag, protocol_version=protocol_version
            )
            events_message = self.sync_engine.datasite_owner_syncer.event_cache.create_events_for_files(
                files
            )
            self.sync_engine.datasite_owner_syncer.queue_event_for_syftbox(
                recipients=[enclave_email],
                file_change_events_message=events_message,
            )
            self.sync_engine.datasite_owner_syncer.process_syftbox_events_queue()

    def _private_share_protocol_version(self, tag: str, peer_email: str) -> str:
        """The protocol version of the copy to ship privately to this peer.

        A reader scans every layout at or below its negotiated version, so the
        newest existing copy at or below it serves; only when none qualifies
        is a copy at the negotiated version materialized.
        """
        storage = self.dataset_manager.storage
        negotiated = storage.negotiated_protocol_version_for_peer(
            peer_email, raise_on_unknown=False
        )
        readable = {
            ref.protocol_version
            for ref in storage.iter_dataset_refs_all_protocols(self.email)
            if ref.name == tag and int(ref.protocol_version) <= int(negotiated)
        }
        if readable:
            return max(readable, key=int)
        self._materialize_dataset_copy(tag, negotiated, users=[peer_email])
        return negotiated

    @property
    def datasets(self) -> Any:
        """The dataset manager. Auto-syncs first unless PRE_SYNC=false."""
        if self.dataset_manager is None:
            raise ValueError("Dataset manager is not set")

        if self._pre_sync_enabled:
            self.sync_engine.sync()

        return self.dataset_manager

    def _resolve_dataset_owners_for_name(self, dataset_name: str) -> list:
        """Resolve which datasite(s) own a dataset of the given name.

        Used by syft.utils.resolve_dataset_files_path when a client is
        passed. Owned here (drives the RDS dataset manager).
        """
        return [
            dataset.owner
            for dataset in self.dataset_manager.get_all()
            if dataset.name == dataset_name
        ]

    def __repr__(self) -> str:
        return f"SyftRDSClient(email={self.sync_engine.email!r})"
