from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from syft_rds import SyftRDSClient, SyftRDSClientConfig
from syft_client.sync.version.peer_manager import CompatAction
from syft_client.sync.peers.peer import Peer
from syft_client.sync.peers.peer_list import PeerList
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_job.job import JobInfo, JobsList
from syft_job.job_storage import JobRef
from syft_job.models import JobState, JobStatus

from syft_enclaves.enclave_job_info import (
    EnclaveJobInfo,
    PartyApprovalStatus,
    enclave_approval_file_name,
)
from syft_enclaves.attestation import (
    AppraisalPolicy,
    verify_attestation_token,
)
from syft_perms.syftperm_context import SyftPermContext

from syft_enclaves.enclave_job_client import EnclaveJobClient
from syft_enclaves.utils import (
    create_clients,
    create_configs,
    setup_callbacks,
    setup_connections,
    wire_peers,
    write_versions,
)
from syft_enclaves.immutability import (
    make_private_dataset_immutability_filter,
)


class SyftEnclaveClient:
    def __init__(
        self,
        rds: SyftRDSClient,
        data_owners: list[str] | None = None,
    ):
        # The Remote Data Science product client. It OWNS the job + dataset
        # surface (job_client / job_runner / dataset_manager) and composes the
        # generic sync engine (a SyftboxManager) as ``rds.sync_engine``.
        self._rds = rds
        # Data owners whose approval gates every job run on this enclave.
        # Fixed at launch/deploy time; stored in memory.
        self.data_owners = list(data_owners or [])

    @property
    def email(self) -> str:
        return self._rds.email

    @property
    def syftbox_folder(self) -> Path:
        return self._rds.syftbox_folder

    @property
    def peers(self) -> PeerList:
        return self._rds.peers

    def add_peer(self, peer_email: str, force: bool = False, verbose: bool = True):
        self._rds.add_peer(peer_email, force=force, verbose=verbose)

    def load_peers(self):
        self._rds.load_peers()

    def approve_peer_request(
        self,
        email_or_peer: str | Peer,
        verbose: bool = True,
        peer_must_exist: bool = True,
    ):
        self._rds.approve_peer_request(
            email_or_peer, verbose=verbose, peer_must_exist=peer_must_exist
        )

    def reject_peer_request(self, email_or_peer: str | Peer):
        self._rds.reject_peer_request(email_or_peer)

    def attest_peer(
        self,
        peer_email: str,
        expected_image_digest: str | None = None,
        policy: "AppraisalPolicy | None" = None,
    ):
        """Verify an enclave peer's attestation by re-reading SYFT_version.json
        from Drive. Returns None (with an info print) when no token is available;
        raises AttestationError only when verification of an existing token fails.

        Args:
            peer_email: the enclave peer to attest.
            expected_image_digest: a "sha256:..." container image digest you
                trust — . When set, the attestation
                is appraised against it.
            policy: a full ``AppraisalPolicy`` for finer control (image digest
                *and* syft-client version). Mutually exclusive with
                ``expected_image_digest``.
        """

        if expected_image_digest is not None and policy is not None:
            raise ValueError("Pass either expected_image_digest or policy, not both.")
        if expected_image_digest is not None:
            policy = AppraisalPolicy(expected_image_digest=expected_image_digest)

        version_info = self._rds.peer_manager.connection_router.read_peer_version_file(
            peer_email
        )
        if version_info is None:
            print(
                f"ℹ️  No version file available for peer {peer_email!r}; skipping attestation."
            )
            return None
        if not version_info.attestation_token:
            print(
                f"ℹ️  Peer {peer_email!r} has no attestation token "
                "(not running in a Confidential Space); skipping attestation."
            )
            return None
        return verify_attestation_token(version_info.attestation_token, policy=policy)

    def sync(self):
        self._rds.sync()

    def delete_syftbox(
        self, verbose: bool = True, broadcast_delete_events: bool = True
    ):
        """Delete all SyftBox state (Drive files + local caches/folder)."""
        self._rds.delete_syftbox(
            verbose=verbose, broadcast_delete_events=broadcast_delete_events
        )

    def create_dataset(self, *args, **kwargs):
        return self._rds.create_dataset(*args, **kwargs)

    def share_private_dataset(self, tag: str, enclave_email: str):
        self._rds.share_private_dataset(tag, enclave_email)

    @property
    def datasets(self) -> SyftDatasetManager:
        return self._rds.dataset_manager

    @property
    def jobs(self) -> JobsList:
        jobs_list = self._rds.job_client.jobs
        wrapped = [
            EnclaveJobInfo.from_job_info(j)
            if j.job_headers.get("job_type") == "enclave"
            else j
            for j in jobs_list
        ]
        return JobsList(wrapped, jobs_list._root_email)

    def submit_python_job(
        self,
        enclave_email: str,
        code_path: str,
        job_name: Optional[str] = "",
        datasets: Optional[dict[str, list[str]]] = None,
        share_results_with_do: bool = False,
        force_submission: bool = False,
        ignore_peer_version: bool = False,
        **kwargs,
    ):
        """Submit a Python job to an enclave, then push files via sync.

        Mirrors ``SyftboxManager.submit_python_job``'s version guard: if the
        enclave's version is unknown/incompatible, raise immediately instead of
        letting ``push_job_files`` silently skip the peer and drop the job (which
        otherwise surfaces much later as a stuck, never-distributed job).
        """
        if not force_submission:
            result = self._rds.peer_manager.get_peer_compatibility_status(
                enclave_email,
                action=CompatAction.SUBMIT,
                ignore_peer_version=ignore_peer_version,
            )
            result.raise_on_skip(operation="submit job")
            result.maybe_warn()
        job_dir = self._rds.job_client.submit_python_job(
            enclave_email,
            code_path,
            job_name,
            datasets=datasets,
            share_results_with_do=share_results_with_do,
            **kwargs,
        )
        self._rds.sync_engine.push_job_files(job_dir)

    def run_jobs(self) -> None:
        """Run approved enclave jobs."""
        for job in self.jobs:
            if (
                job.status == "approved"
                and job.job_headers.get("job_type") == "enclave"
            ):
                state = JobState.load(job.job_review_path / "state.yaml")
                if state.status != JobStatus.APPROVED:
                    state.status = JobStatus.APPROVED
                    state.save(job.job_review_path / "state.yaml")

        self._rds.process_approved_jobs(
            force_execution=True,
            share_outputs_with_submitter=True,
            share_logs_with_submitter=True,
        )

    def distribute_results(self) -> None:
        """Distribute job results to DS (always) and optionally to DOs."""
        for job in self.jobs:
            if job.status != "done":
                continue
            results_shared_marker = job.job_review_path / "results_shared"
            if results_shared_marker.exists():
                continue

            # Always share results with the DS (submitter)
            self._forward_results_to_recipients(job, [job.submitted_by])

            # Optionally share with DOs
            if job.job_headers.get("share_results_with_do"):
                datasets = job.job_metadata.datasets
                if datasets:
                    do_emails = list(datasets.keys())
                    job.share_outputs(do_emails)
                    self._forward_results_to_recipients(job, do_emails)

            results_shared_marker.write_text("shared")

        self._rds.sync()

    def _read_state_file(self, job: JobInfo) -> dict[Path, bytes]:
        """Read the job state.yaml as a {path_in_datasite: bytes} dict."""
        state_file = job.job_review_path / "state.yaml"
        if not state_file.exists():
            return {}
        datasite_dir = self._rds.syftbox_folder / self._rds.email
        state_rel = state_file.relative_to(datasite_dir)
        return {state_rel: state_file.read_bytes()}

    def _forward_results_to_recipients(self, job: JobInfo, recipients: list[str]):
        """Forward job output files and state to recipients via event outbox."""
        outputs_dir = job.job_review_path / "outputs"
        if not outputs_dir.exists():
            return
        files_by_datasite_path = self._get_files_in_dir(outputs_dir)
        files_by_datasite_path.update(self._read_state_file(job))
        if not files_by_datasite_path:
            return
        syncer = self._rds.sync_engine.datasite_owner_syncer
        events_message = syncer.event_cache.create_events_for_files(
            files_by_datasite_path
        )
        syncer.queue_event_for_syftbox(
            recipients=recipients,
            file_change_events_message=events_message,
        )
        syncer.process_syftbox_events_queue()

    def approve_job(self, job: JobInfo) -> None:
        """Approve an enclave job and push the approval state file to the enclave."""
        job.approve()
        file_name = enclave_approval_file_name(self.email)
        approval_file = job.job_review_path / file_name
        relative_path = approval_file.relative_to(self._rds.syftbox_folder)
        self._rds.sync_engine.datasite_watcher_syncer.on_file_change(
            relative_path, process_now=True
        )

    def receive_jobs(self):
        """Receive and distribute enclave jobs to relevant DOs.

        1. Scans inbox for new submissions
        2. For enclave jobs (with datasets), forwards files to each DO
        3. Creates JobState with PartyApprovalStatus per DO
        4. Sets permissions and marks as distributed
        """
        self._rds.job_client.scan_inbox()
        job_manager = self._rds.job_client.manager
        for ref in job_manager.iter_submission_refs(self._rds.email):
            self._try_distribute_job(ref)

    def _try_distribute_job(self, ref: JobRef):
        """Distribute a single enclave job to relevant DOs if not yet distributed."""
        job_manager = self._rds.job_client.manager
        job_dir = job_manager.submission_dir(ref)
        config = job_manager.read_submission(ref)
        if config.job_type != "enclave" or not config.datasets:
            return

        review_dir = job_manager.review_dir(ref)
        distributed_marker = review_dir / "distributed"
        if distributed_marker.exists():
            return

        # Forward the job to the DOs referenced in the submission, but gate
        # approval on the enclave's globally-configured data owners. The job is
        # forwarded to the union so every required approver can review it.
        # Jobs may reference datasets hosted on the enclave's own datasite
        # (e.g. logs it collects) — never forward to ourselves.
        submission_dos = [e for e in config.datasets.keys() if e != self.email]
        approval_dos = self.data_owners
        recipients = list(dict.fromkeys([*submission_dos, *approval_dos]))
        self._forward_job_to_dos(job_dir, recipients)
        self._save_enclave_job_state(review_dir, approval_dos, config.datasets)
        self._set_job_permissions(ref, recipients, approval_dos)
        self._forward_approval_files_to_dos(review_dir, approval_dos)

        distributed_marker.parent.mkdir(parents=True, exist_ok=True)
        distributed_marker.write_text("distributed")

    def _forward_job_to_dos(self, job_dir: Path, do_emails: list[str]):
        """Forward job files to DOs via the event-based outbox mechanism."""
        files_by_datasite_path = self._get_files_in_dir(job_dir)
        syncer = self._rds.sync_engine.datasite_owner_syncer
        events_message = syncer.event_cache.create_events_for_files(
            files_by_datasite_path
        )
        syncer.queue_event_for_syftbox(
            recipients=do_emails,
            file_change_events_message=events_message,
        )
        syncer.process_syftbox_events_queue()

    def _forward_approval_files_to_dos(self, review_dir: Path, do_emails: list[str]):
        """Forward each DO's approval state file to them individually."""
        datasite_dir = self._rds.syftbox_folder / self._rds.email
        syncer = self._rds.sync_engine.datasite_owner_syncer
        for do_email in do_emails:
            file_name = enclave_approval_file_name(do_email)
            approval_file = review_dir / file_name
            path_in_datasite = approval_file.relative_to(datasite_dir)
            files_by_datasite_path = {path_in_datasite: approval_file.read_bytes()}
            events_message = syncer.event_cache.create_events_for_files(
                files_by_datasite_path
            )
            syncer.queue_event_for_syftbox(
                recipients=[do_email],
                file_change_events_message=events_message,
            )
            syncer.process_syftbox_events_queue()

    def _get_files_in_dir(self, directory: Path) -> dict[Path, bytes]:
        """Read all files under directory, keyed by path relative to the datasite root."""
        datasite_dir = self._rds.syftbox_folder / self._rds.email
        files_by_datasite_path = {}
        for f in directory.rglob("*"):
            if not f.is_file():
                continue
            path_in_datasite = f.relative_to(datasite_dir)
            files_by_datasite_path[Path(path_in_datasite)] = f.read_bytes()
        return files_by_datasite_path

    def _save_enclave_job_state(
        self,
        review_dir: Path,
        do_emails: list[str],
        datasets: dict[str, list[str]],
    ):
        """Create JobState and individual <do_email>_approval_state.json files."""
        state = JobState(
            status=JobStatus.PENDING,
            received_at=datetime.now(timezone.utc),
        )
        review_dir.mkdir(parents=True, exist_ok=True)
        state.save(review_dir / "state.yaml")

        for do_email in do_emails:
            approval = PartyApprovalStatus(
                party=do_email,
                dataset=",".join(datasets.get(do_email, [])),
            )
            approval.save_json(review_dir / enclave_approval_file_name(do_email))

    def _set_job_permissions(
        self,
        ref: JobRef,
        read_dos: list[str],
        approval_dos: list[str],
    ):
        """Grant inbox read to everyone who needs to see the job (referenced +
        approving DOs), and approval-file write to the approving DOs."""
        job_manager = self._rds.job_client.manager
        datasite = self._rds.syftbox_folder / self._rds.email
        ctx = SyftPermContext(datasite=datasite)
        inbox_rel = job_manager.submission_dir(ref).relative_to(datasite)
        review_rel = job_manager.review_dir(ref).relative_to(datasite)

        for do_email in dict.fromkeys([*read_dos, *approval_dos]):
            ctx.open(inbox_rel).grant_read_access(do_email)

        for do_email in approval_dos:
            approval_rel = review_rel / enclave_approval_file_name(do_email)
            ctx.open(approval_rel).grant_write_access(do_email)

    @classmethod
    def from_config(
        cls,
        config: SyftRDSClientConfig,
        data_owners: list[str] | None = None,
    ) -> "SyftEnclaveClient":
        """Build a SyftEnclaveClient from an RDS config with a wrapped job_client.

        The enclave client composes a ``SyftRDSClient`` (the RDS product client),
        which OWNS the job + dataset surface and holds the generic sync engine at
        ``rds.sync_engine``. We wrap the RDS-owned job_client with
        ``EnclaveJobClient`` (settable on the RDS client) and install the private
        dataset immutability filter on the nested sync engine's watcher cache.

        Encryption rides along on ``config.sync`` via
        ``peer_manager_config.use_encryption``; ``SyftboxManager.from_config``
        (invoked inside ``SyftRDSClient.from_config``) loads/generates this
        datasite's keys from its per-datasite key file, so no extra step is needed.
        """
        rds = SyftRDSClient.from_config(config)
        rds.job_client = EnclaveJobClient(rds.job_client)

        sync_engine = rds.sync_engine
        if sync_engine.datasite_watcher_syncer:
            sync_engine.datasite_watcher_syncer.datasite_watcher_cache.pre_write_filter = make_private_dataset_immutability_filter(
                sync_engine.syftbox_folder
            )

        return cls(rds, data_owners=data_owners)

    @classmethod
    def for_enclave(
        cls,
        email: str,
        token_path: Path | str | None = None,
        data_owners: list[str] | None = None,
        encryption: bool = False,
    ) -> "SyftEnclaveClient":
        """Build an enclave client backed by a real Google Drive connection.
        Args:
            email: The enclave datasite's email address.
            token_path: Path to a pre-authorized Google Drive OAuth token.
            data_owners: Emails whose approval gates every job on this enclave.
            encryption: Enable end-to-end drive encryption.
        """
        config = SyftRDSClientConfig.for_jupyter(
            email=email,
            has_ds_role=True,
            has_do_role=True,
            token_path=Path(token_path) if token_path is not None else None,
            encryption=encryption,
        )

        # Note: We do not currently provide the ability to load encryption keys passed during creation of enclave.
        # To be able to support loading existing enclave private keys, we need to have Key release flow , like we have for releasing
        # token.json for enclaves, currently each time an enclave boots up, we get a fresh key pair.
        return cls.from_config(config, data_owners=data_owners)

    @classmethod
    def quad_with_mock_drive_service_connection(
        cls,
        enclave_email: str | None = None,
        do1_email: str | None = None,
        do2_email: str | None = None,
        ds_email: str | None = None,
        use_in_memory_cache: bool = True,
        encryption: bool = False,
    ) -> tuple[
        "SyftEnclaveClient",
        "SyftEnclaveClient",
        "SyftEnclaveClient",
        "SyftEnclaveClient",
    ]:
        """Create 4 interconnected clients for testing enclave scenarios.

        Peer topology:
        - Enclave: peers with DO1, DO2, DS
        - DO1: peers with DS, Enclave (not DO2)
        - DO2: peers with DS, Enclave (not DO1)
        - DS: peers with DO1, DO2, Enclave

        Args:
            encryption: Enable end-to-end drive encryption on all four clients.
        Returns:
            Tuple of (enclave, do1, do2, ds)
        """
        configs = create_configs(
            enclave_email, do1_email, do2_email, ds_email, use_in_memory_cache
        )
        clients = create_clients(configs)
        enclave, do1, do2, ds = clients
        enclave.data_owners = [do1.email, do2.email]
        # The setup helpers manipulate the generic sync engine directly
        # (connections, file-watcher callbacks, version files, encryption,
        # peering). Each client's sync engine lives at ``rds.sync_engine``; the
        # RDS-owned job/dataset managers react via the peer-lifecycle callbacks
        # registered on that same engine during SyftRDSClient.__init__.
        managers = tuple(c._rds.sync_engine for c in clients)
        setup_connections(managers)
        setup_callbacks(managers)
        write_versions(managers)
        # Initialize encryption AFTER connections/versions but BEFORE peering,
        # so the bundle exchange in wire_peers() picks up each client's keys
        # (mirrors SyftboxManager.pair_with_mock_drive_service_connection).
        if encryption:
            for manager in managers:
                manager._init_encrypted_peer_store()
        wire_peers(managers)

        return clients
