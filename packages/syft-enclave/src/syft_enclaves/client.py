import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from syft_datasets import Dataset
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_job.read_permissions import (
    OUTPUTS,
    ReadPermissions,
    merge_read_permissions,
)
from syft_job.job import JobInfo, JobsList
from syft_job.job_storage import JobRef
from syft_job.models import JobStatus
from syft_perms.syftperm_context import SyftPermContext
from syft_rds import SyftRDSClient, SyftRDSClientConfig
from syft_rds.waiting import JobCheck, JobStatusArg, wait_for_job

from syft.sync.connections.protocols import Connections
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.peers.peer import Peer
from syft.sync.peers.peer_list import PeerList
from syft.sync.utils.waiting import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    wait_for,
)
from syft.sync.version.peer_manager import CompatAction
from syft_enclaves.attestation.dispatch import policy_for, verify_evidence
from syft_enclaves.attestation.envelope import AttestationEvidence
from syft_enclaves.enclave_job_client import EnclaveJobClient
from syft_enclaves.enclave_job_info import (
    EnclaveJobInfo,
    PartyApprovalStatus,
    enclave_approval_file_name,
)
from syft_enclaves.immutability import (
    make_private_dataset_immutability_filter,
)
from syft_enclaves.log_requests import (
    log_request_file_name,
    logs_requested_by_all_parties,
    write_log_head,
)
from syft_enclaves.receipt.writer import (
    ReceiptSettings,
    mark_started,
    write_receipt,
    write_receipt_error,
)
from syft_enclaves.utils import (
    create_clients,
    create_configs,
    setup_callbacks,
    wire_peers,
    write_versions,
)

RESULTS_SHARED_MARKER = "results_shared"
# How the enclave records an approval: every data owner approved the same job.
ENCLAVE_APPROVAL_METHOD = "enclave"


def pre_sync_enabled() -> bool:
    """Whether the client syncs on its own. ``PRE_SYNC=false`` turns it off."""
    return os.environ.get("PRE_SYNC", "true").lower() == "true"


if TYPE_CHECKING:
    # Only for the attest_peer annotation: importing the tinfoil policy at
    # runtime would drag the optional SDK onto the always-imported path.
    from syft_enclaves.attestation import AppraisalPolicy
    from syft_enclaves.attestation.tinfoil import TinfoilAppraisalPolicy

logger = logging.getLogger(__name__)


def _given_shorthands(
    expected_image_digest: str | None,
    expected_data_owners: list[str] | None,
    expected_email: str | None,
    policy: "AppraisalPolicy | TinfoilAppraisalPolicy | None",
) -> dict:
    """The shorthand arguments of ``attest_peer`` that were given.

    Raises:
        ValueError: a shorthand and ``policy`` were both given.
    """
    shorthands = {
        "expected_image_digest": expected_image_digest,
        "expected_data_owners": expected_data_owners,
        "expected_email": expected_email,
    }
    given = {name: value for name, value in shorthands.items() if value is not None}
    if given and policy is not None:
        raise ValueError(f"Pass either {' / '.join(shorthands)} or policy, not both.")
    return given


class SyftEnclaveClient:
    def __init__(
        self,
        rds: SyftRDSClient,
        data_owners: list[str] | None = None,
        receipts: ReceiptSettings | None = None,
    ):
        # The Remote Data Science product client. It OWNS the job + dataset
        # surface (job_client / job_runner / dataset_manager) and composes the
        # generic sync engine (a SyftboxManager) as ``rds.sync_engine``.
        self._rds = rds
        # Data owners whose approval gates every job run on this enclave.
        # Fixed at launch/deploy time; stored in memory.
        self.data_owners = list(data_owners or [])
        # When set, every finished job gets a signed receipt in its outputs.
        self.receipts = receipts

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

    def validate_peer(self, peer_email: str) -> Peer:
        return self._rds.validate_peer(peer_email)

    def attest_peer(
        self,
        peer_email: str,
        expected_image_digest: str | None = None,
        expected_data_owners: list[str] | None = None,
        expected_email: str | None = None,
        policy: "AppraisalPolicy | TinfoilAppraisalPolicy | None" = None,
    ):
        """Verify an enclave peer's attestation by re-reading SYFT_version.json
        from Drive. Returns None (with an info print) when the peer published no
        evidence; raises AttestationError when verification of existing evidence
        fails.

        The peer's evidence says which TEE produced it, and it is routed to that
        target's verifier (Confidential Space or Tinfoil). Verifying Tinfoil
        evidence needs the optional ``tinfoil`` package; see
        ``docs/tinfoil_deployment.md``.

        A policy has to pin an image digest, a data-owner list and the enclave's
        email, so pass all three shorthands or build a policy yourself. Without
        them the attestation would prove that some genuine enclave exists, but
        not which code it runs, which datasite it runs as, or who approves a
        job on it. To accept that on purpose, pass a policy with
        ``allow_unpinned=True``.

        Args:
            peer_email: the enclave peer to attest.
            expected_image_digest: a "sha256:..." container image digest you
                trust.
            expected_data_owners: the emails whose approval must gate a job on
                this enclave.
            expected_email: the datasite the enclave should be running as.
            policy: a full appraisal policy for finer control — an
                ``AppraisalPolicy`` for Confidential Space or a
                ``TinfoilAppraisalPolicy`` for Tinfoil. Mutually exclusive with
                the shorthands.
        """

        given = _given_shorthands(
            expected_image_digest, expected_data_owners, expected_email, policy
        )
        evidence = self._peer_evidence(peer_email)
        if evidence is None:
            return None
        if given:
            policy = policy_for(evidence.kind, **given)
        result = verify_evidence(evidence, policy=policy)
        self._adopt_verified_key_bundle(peer_email, result)
        return result

    def _adopt_verified_key_bundle(self, peer_email: str, result) -> None:
        """Trust the peer's keys when attestation bound them to its report.

        Only a bundle delivered over a channel pinned to the attested TLS key
        gets here — see ``attestation.https``. The copy the peer publishes to
        Drive is unsigned, so a mismatch means the Drive copy was tampered
        with; the bound one wins and we say so rather than failing, since the
        bound one is exactly what we should be using.
        """
        bundle = getattr(result, "verified_key_bundle", None)
        if not bundle:
            return
        peer_store = self._rds.peer_manager.peer_store
        previous = None
        if peer_store.has_peer_bundle(peer_email):
            previous = peer_store.get_cached_peer(peer_email).public_encryption_bundle
        if previous and previous != bundle:
            print(
                f"⚠️  {peer_email!r} published a different key bundle to Drive than "
                "the one its attestation binds; using the attested one."
            )
        peer_store.set_peer_bundle(peer_email, bundle)
        self._persist_peer_bundle(peer_email, bundle)
        print(f"🔑 Set {peer_email!r} encryption keys from its attestation.")

    def _persist_peer_bundle(self, peer_email: str, bundle: dict) -> None:
        """Write the bundle alongside the peer's state, as load_peers does."""
        peer = self._rds.peer_manager.peer_store.get_cached_peer(peer_email)
        if peer is None:
            return
        self._rds.peer_manager.connection_router.update_peer_state(
            peer_email, peer.state.value, public_encryption_bundle=bundle
        )

    def wait_until_attested(
        self,
        peer_email: str,
        expected_image_digest: str | None = None,
        expected_data_owners: list[str] | None = None,
        expected_email: str | None = None,
        policy: "AppraisalPolicy | TinfoilAppraisalPolicy | None" = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        """Wait until ``peer_email`` publishes attestation evidence, then verify it.

        The arguments and the result are those of ``attest_peer``. Use it only
        for an enclave that runs in a TEE: an enclave outside a TEE never
        publishes evidence, so the wait ends in ``TimeoutError``.

        Raises:
            TimeoutError: no evidence was published within ``timeout`` seconds.
            AttestationError: the evidence fails verification.
        """
        _given_shorthands(
            expected_image_digest, expected_data_owners, expected_email, policy
        )
        wait_for(
            lambda: self._peer_evidence(peer_email, quiet=True),
            f"the attestation of {peer_email}",
            lambda: "no attestation evidence published yet",
            timeout,
            poll_interval,
        )
        return self.attest_peer(
            peer_email,
            expected_image_digest=expected_image_digest,
            expected_data_owners=expected_data_owners,
            expected_email=expected_email,
            policy=policy,
        )

    def _peer_evidence(
        self, peer_email: str, quiet: bool = False
    ) -> "AttestationEvidence | None":
        """The peer's published attestation evidence, or None if it has none.

        Prints why there is none, unless ``quiet`` is set.
        """
        peer_info = self._rds.peer_manager.connection_router.read_peer_info_file(
            peer_email
        )
        if peer_info is None:
            if not quiet:
                print(
                    f"ℹ️  No version file available for peer {peer_email!r}; skipping attestation."
                )
            return None
        # A malformed envelope raises rather than skipping: a peer that
        # published something unparseable is not the same as one that
        # published nothing.
        evidence = AttestationEvidence.read_from(peer_info.version_info)
        if evidence is None and not quiet:
            print(
                f"ℹ️  Peer {peer_email!r} published no attestation evidence "
                "(not running in an attested enclave); skipping attestation."
            )
        return evidence

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
        return self._wrap_enclave_jobs(self._rds.jobs)

    def _wrap_enclave_jobs(self, jobs_list: JobsList) -> JobsList:
        """``jobs_list`` with every enclave job wrapped by ``_as_enclave_job``."""
        wrapped = [
            self._as_enclave_job(j) if j.job_headers.get("job_type") == "enclave" else j
            for j in jobs_list
        ]
        return JobsList(wrapped, jobs_list._root_email, jobs_list._has_do_role)

    def wait_until_peered(
        self,
        peer_email: str,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> Peer:
        """Same as ``SyftRDSClient.wait_until_peered``."""
        return self._rds.wait_until_peered(
            peer_email, timeout=timeout, poll_interval=poll_interval
        )

    def wait_until_has_dataset(
        self,
        name: str,
        datasite: str | None = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> Dataset:
        """Same as ``SyftRDSClient.wait_until_has_dataset``."""
        return self._rds.wait_until_has_dataset(
            name, datasite=datasite, timeout=timeout, poll_interval=poll_interval
        )

    def wait_until_has_job(
        self,
        job_name: str,
        user_name: str | None = None,
        status: JobStatusArg = None,
        where: JobCheck | None = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> JobInfo:
        """Like ``SyftRDSClient.wait_until_has_job``, with enclave job statuses.

        An enclave job is read as in ``self.jobs``, so its status comes from
        the approval files of the data owners.
        """
        return wait_for_job(
            self.sync,
            lambda: self._wrap_enclave_jobs(self._local_jobs()),
            job_name,
            user_name,
            status,
            where,
            timeout,
            poll_interval,
        )

    def _as_enclave_job(self, job: JobInfo) -> EnclaveJobInfo:
        """Wrap ``job``; on the enclave that runs it, every data owner must approve.

        The configured data owners are the approvers, as when the job was
        distributed, so the enclave never counts only the approval files that
        happen to exist.
        """
        runs_here = job.datasite_owner_email == self.email
        required = list(self.data_owners) if runs_here else None
        return EnclaveJobInfo.from_job_info(
            job,
            required_approvers=required,
            on_file_change=self._push_file,
        )

    def submit_python_job(
        self,
        enclave_email: str,
        code_path: str,
        job_name: Optional[str] = "",
        datasets: Optional[dict[str, list[str]]] = None,
        share_logs_with_submitter: bool = False,
        share_outputs_with_submitter: bool = False,
        share_logs_with_owners: bool = False,
        share_outputs_with_owners: bool = False,
        share_files: Optional[ReadPermissions] = None,
        force_submission: bool = False,
        ignore_peer_version: bool = False,
        **kwargs,
    ):
        """Submit a Python job to an enclave, then push files via sync.

        The ``*_with_submitter`` flags ask for stdout+stderr / the outputs for
        the submitter, the ``*_with_owners`` flags for every data owner in
        ``datasets``. ``share_files`` gives the request as {file: [emails]}
        instead of the flags. Every party must approve a file for an email.

        Raises:
            ValueError: If share_files comes with a share flag.
        """
        if not force_submission:
            self._check_can_submit_to(enclave_email, ignore_peer_version)
        job_dir = self._rds.job_client.submit_python_job(
            enclave_email,
            code_path,
            job_name,
            datasets=datasets,
            share_logs_with_submitter=share_logs_with_submitter,
            share_outputs_with_submitter=share_outputs_with_submitter,
            share_logs_with_owners=share_logs_with_owners,
            share_outputs_with_owners=share_outputs_with_owners,
            share_files=share_files,
            **kwargs,
        )
        self._rds.sync_engine.push_job_files(job_dir)

    def _check_can_submit_to(self, enclave_email: str, ignore_peer_version: bool):
        """Raise when the enclave's version is unknown or incompatible.

        Mirrors ``SyftboxManager.submit_python_job``'s version guard, so a job
        is not silently skipped by ``push_job_files`` and never distributed.
        """
        result = self._rds.peer_manager.get_peer_compatibility_status(
            enclave_email,
            action=CompatAction.SUBMIT,
            ignore_peer_version=ignore_peer_version,
        )
        result.raise_on_skip(operation="submit job")
        result.maybe_warn()

    def run_jobs(self) -> None:
        """Run approved enclave jobs.

        A job counts as approved when every data owner approved this same
        submission. The enclave records it as approved, and the runner executes
        a fresh copy only while it still matches.
        """
        for job in self.jobs:
            if (
                job.status == "approved"
                and job.job_headers.get("job_type") == "enclave"
            ):
                self._mark_approved_job_started(job)

        # share_logs_with_submitter=False shares the logs with nobody. Only an
        # approval from every party shares them. With receipts on, outputs
        # reach the submitter only through distribute_results, after the
        # receipt is written, so results and receipt arrive together instead
        # of the receipt a sync later.
        self._rds.process_approved_jobs(
            force_execution=True,
            share_logs_with_submitter=False,
            share_outputs_with_submitter=self.receipts is None,
        )

    def _mark_approved_job_started(self, job: JobInfo) -> None:
        """Record that every data owner approved ``job``, so the runner executes it."""
        try:
            job.record_approval(approval_method=ENCLAVE_APPROVAL_METHOD)
        except ValueError as e:
            logger.warning(f"Not running enclave job '{job.name}': {e}")
            return
        manager = self._rds.job_client.manager
        state = manager.read_state(job._ref)
        if state.status != JobStatus.APPROVED:
            state.status = JobStatus.APPROVED
            manager.write_state(job._ref, state)
        mark_started(job.job_review_path, job.job_submission_path)

    def read_permissions_to_share(self, job: JobInfo) -> ReadPermissions:
        """Per file, the emails the submitter requested and every party approved."""
        return self._as_enclave_job(job).read_permissions_to_share

    def _local_jobs(self) -> JobsList:
        """The job list read from disk, with no sync.

        ``self.jobs`` syncs first, therefore a caller that must act before a
        push reads the list through this method.
        """
        return self._rds.job_client.jobs

    def _share_requested_logs(self, job: JobInfo) -> ReadPermissions:
        """Share each log that every data owner requested the same, after the run.

        Returns what it shared: the log, or a copy of its first chars.
        """
        review = job.job_review_path
        shared = {}
        requests = logs_requested_by_all_parties(review, self.data_owners)
        for log_file, request in requests.items():
            name = write_log_head(review, log_file, request.n_chars)
            if name is not None:
                shared[name] = request.readers or list(self.data_owners)
        job.share(shared)
        return shared

    def _record_approved_read_permissions(
        self, ref: JobRef, approved: ReadPermissions
    ) -> None:
        """Write what every party approved into the job's state, when it changed."""
        manager = self._rds.job_client.manager
        state = manager.read_state(ref)
        if state.approved_read_permissions != approved:
            state.approved_read_permissions = approved
            manager.write_state(ref, state)

    def distribute_results(self) -> None:
        """Share each finished job's results and requested logs, then sync.

        Runs on every tick, because an approval or a log request can come after
        the job finished.
        """
        for job in self.jobs:
            if job.status not in ("done", "failed"):
                continue
            approved = self.read_permissions_to_share(job)
            self._share_results(job, approved)
            if job.job_headers.get("job_type") == "enclave":
                shared_logs = self._share_requested_logs(job)
                self._record_approved_read_permissions(
                    job._ref, merge_read_permissions(approved, shared_logs)
                )

        self._rds.sync()

    def _share_results(self, job: JobInfo, approved: ReadPermissions) -> None:
        """Share each result that the submitter requested and every party approved.

        The submitter always gets the outputs.
        """
        self._write_receipt_once(job)
        output_readers = [job.submitted_by, *approved.get(OUTPUTS, [])]
        job.share({**approved, OUTPUTS: list(dict.fromkeys(output_readers))})

    def _write_receipt_once(self, job: JobInfo) -> None:
        """Sign a receipt into the outputs, before they first go out."""
        marker = job.job_review_path / RESULTS_SHARED_MARKER
        if marker.exists():
            return
        if self.receipts is not None and job.status == "done":
            self._try_write_receipt(job)
        marker.write_text("shared")

    def _try_write_receipt(self, job: JobInfo) -> None:
        """Sign a receipt into the job's outputs, so it ships with them.

        A failure is not raised: the results still go out, with the error in
        place of the receipt, rather than being held back on every tick.
        """
        try:
            write_receipt(self, job, self.receipts)
        except Exception:
            logger.exception("Could not write a receipt for job %s", job.name)
            write_receipt_error(job)

    def _push_file(self, changed_file: Path) -> None:
        """Push a changed file to the enclave."""
        relative_path = changed_file.relative_to(self._rds.syftbox_folder)
        self._rds.sync_engine.datasite_watcher_syncer.on_file_change(
            relative_path, process_now=True
        )

    def _require_enclave_job(self, job: JobInfo) -> EnclaveJobInfo:
        """Wrap ``job`` like ``self.jobs`` does, and refuse a job of another kind."""
        if not isinstance(job, EnclaveJobInfo):
            raise TypeError(
                f"Job '{job.name}' is not an enclave job, so it carries no "
                f"read permissions. Approve it through the datasite client."
            )
        return self._as_enclave_job(job)

    def approve_job(
        self,
        job: JobInfo,
        share_files: Optional[ReadPermissions] = None,
        share_requested_files: Optional[bool] = None,
    ) -> None:
        """Approve an enclave job and push the approval state file to the enclave.

        Same as ``job.approve(...)`` on a job from ``self.jobs``: with no
        share_files, it approves everything the submitter requested.
        """
        job = self._require_enclave_job(job)
        if pre_sync_enabled():
            self._rds.sync()
        job.approve(
            share_files=share_files,
            share_requested_files=share_requested_files,
        )

    def reject_job(self, job: JobInfo, reason: Optional[str] = None) -> None:
        """Reject an enclave job and push the approval state file to the enclave.

        Same as ``job.reject(reason)`` on a job from ``self.jobs``. It also
        withdraws an earlier approval.
        """
        job = self._require_enclave_job(job)
        if pre_sync_enabled():
            self._rds.sync()
        job.reject(reason)

    def update_read_permissions(
        self, job: JobInfo, read_permissions: Optional[ReadPermissions]
    ) -> ReadPermissions:
        """Change who this data owner lets read which file, and push the change.

        Same as ``job.update_read_permissions(...)`` on a job from ``self.jobs``.
        """
        job = self._require_enclave_job(job)
        if pre_sync_enabled():
            self._rds.sync()
        return job.update_read_permissions(read_permissions)

    def request_stderr(
        self,
        job: JobInfo,
        readers: str | list[str] | None = None,
        n_chars: Optional[int] = None,
    ) -> None:
        """Ask the enclave to share the job's stderr, or only its first ``n_chars`` chars.

        Only a data owner can ask. The enclave shares it once every data owner
        made the same request. ``readers`` None means the data owners this
        enclave is configured with.
        """
        self._request_log(job, "stderr.txt", readers, n_chars)

    def request_stdout(
        self,
        job: JobInfo,
        readers: str | list[str] | None = None,
        n_chars: Optional[int] = None,
    ) -> None:
        """Same as ``request_stderr``, for stdout."""
        self._request_log(job, "stdout.txt", readers, n_chars)

    def _request_log(
        self,
        job: JobInfo,
        log_file: str,
        readers: str | list[str] | None,
        n_chars: Optional[int],
    ) -> None:
        job = self._require_enclave_job(job)
        if pre_sync_enabled():
            self._rds.sync()
        job.request_log(log_file, readers, n_chars)

    def receive_jobs(self):
        """Receive and distribute enclave jobs to relevant DOs.

        1. Scans inbox for new submissions
        2. For enclave jobs (with datasets), forwards files to each DO
        3. Writes a PartyApprovalStatus file per DO
        4. Sets permissions and marks as distributed
        """
        self._rds.job_client.scan_inbox()
        job_manager = self._rds.job_client.manager
        for ref in job_manager.iter_submission_refs(self._rds.email):
            self._try_distribute_job(ref)

    def _try_distribute_job(self, ref: JobRef):
        """Distribute a single enclave job to relevant DOs if not yet distributed.

        Only a received job is distributed: one whose files all arrived and
        whose hash the enclave recorded. A job still arriving, or one rejected
        at receipt, is not.
        """
        job_manager = self._rds.job_client.manager
        if job_manager.read_owner_record(ref) is None:
            return
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
        self._write_approval_files(review_dir, approval_dos, config.datasets)
        self._set_job_permissions(ref, recipients, approval_dos)
        self._forward_review_files_to_dos(review_dir, approval_dos)

        distributed_marker.parent.mkdir(parents=True, exist_ok=True)
        distributed_marker.write_text("distributed")

    def _forward_job_to_dos(self, job_dir: Path, do_emails: list[str]):
        """Forward job files to DOs via the event-based outbox mechanism."""
        files_by_datasite_path = self._get_files_in_dir(job_dir)
        syncer = self._rds.sync_engine.datasite_owner_syncer
        events_message = syncer.event_cache.create_events_for_files(
            files_by_datasite_path
        )
        syncer.process_events_for_own_state_and_peers(
            recipients=do_emails,
            file_change_events_message=events_message,
        )

    def _forward_review_files_to_dos(self, review_dir: Path, do_emails: list[str]):
        """Forward each DO its approval state file, with the job's state.

        The state holds the submitter's request, which a DO approves within.
        """
        datasite_dir = self._rds.syftbox_folder / self._rds.email
        syncer = self._rds.sync_engine.datasite_owner_syncer
        state_file = review_dir / "state.yaml"
        for do_email in do_emails:
            approval_file = review_dir / enclave_approval_file_name(do_email)
            files_by_datasite_path = {
                f.relative_to(datasite_dir): f.read_bytes()
                for f in (approval_file, state_file)
            }
            events_message = syncer.event_cache.create_events_for_files(
                files_by_datasite_path
            )
            syncer.process_events_for_own_state_and_peers(
                recipients=[do_email],
                file_change_events_message=events_message,
            )

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

    def _write_approval_files(
        self,
        review_dir: Path,
        do_emails: list[str],
        datasets: dict[str, list[str]],
    ):
        """Write a pending <do_email>_approval_state.json file per DO.

        ``scan_inbox`` already wrote the pending state, with the submitter's request.
        """
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
        approving DOs), state read to them and the submitter, and write to each
        approving DO on its own approval and log request files."""
        job_manager = self._rds.job_client.manager
        datasite = self._rds.syftbox_folder / self._rds.email
        ctx = SyftPermContext(datasite=datasite)
        inbox_rel = job_manager.submission_dir(ref).relative_to(datasite)
        review_rel = job_manager.review_dir(ref).relative_to(datasite)

        for do_email in dict.fromkeys([*read_dos, *approval_dos]):
            ctx.open(inbox_rel).grant_read_access(do_email)

        # A rule for one file replaces the folder rule that lets the submitter
        # read the state, so this rule names the submitter too.
        for email in dict.fromkeys([ref.ds_email, *read_dos, *approval_dos]):
            ctx.open(review_rel / "state.yaml").grant_read_access(email)

        for do_email in approval_dos:
            for name in (
                enclave_approval_file_name(do_email),
                log_request_file_name(do_email),
            ):
                ctx.open(review_rel / name).grant_write_access(do_email)

    @classmethod
    def from_config(
        cls,
        config: SyftRDSClientConfig,
        data_owners: list[str] | None = None,
        connections: Connections | None = None,
    ) -> "SyftEnclaveClient":
        """Build a SyftEnclaveClient from an RDS config with a wrapped job_client.

        Encryption rides along on ``config.sync`` via
        ``peer_manager_config.encryption_policy``; ``SyftboxManager.from_config``
        (invoked inside ``SyftRDSClient.from_config``) loads/generates this
        datasite's keys from its per-datasite key file, so no extra step is needed.
        ``connections`` replace the ones ``config.sync`` describes; tests pass
        them over a mock Drive.
        """
        rds = SyftRDSClient.from_config(config, connections)
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
        persist_owner_state: bool = False,
    ) -> "SyftEnclaveClient":
        """Build an enclave client backed by a real Google Drive connection.

        An enclave always requires encryption with every peer.

        Args:
            email: The enclave datasite's email address.
            token_path: Path to a pre-authorized Google Drive OAuth token.
            data_owners: Emails whose approval gates every job on this enclave.
            persist_owner_state: Keep the owner-only state used to restore
                this datasite later (event log, rolling state, checkpoints).
                Off by default - an enclave gets ephemeral keys and wipes its
                state on boot, so there is never anything to restore.
        """
        config = SyftRDSClientConfig.for_jupyter(
            email=email,
            has_ds_role=True,
            has_do_role=True,
            token_path=Path(token_path) if token_path is not None else None,
            encryption=True,
            persist_owner_state=persist_owner_state,
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
        enclave_persist_owner_state: bool = False,
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
            encryption: Policy of the DOs and the DS. The enclave always
                requires encryption.
            enclave_persist_owner_state: Whether the enclave keeps restorable
                owner state. Off by default, as in production.
        Returns:
            Tuple of (enclave, do1, do2, ds)
        """
        policies = [EncryptionPolicy.REQUIRED] + [
            EncryptionPolicy.from_bool(encryption)
        ] * 3
        configs = create_configs(
            enclave_email,
            do1_email,
            do2_email,
            ds_email,
            use_in_memory_cache,
            policies,
            enclave_persist_owner_state,
        )
        clients = create_clients(configs)
        enclave, do1, do2, ds = clients
        enclave.data_owners = [do1.email, do2.email]
        # The setup helpers manipulate the generic sync engine directly
        # (file-watcher callbacks, version files, encryption, peering). Each client's sync engine lives at ``rds.sync_engine``; the
        # RDS-owned job/dataset managers react via the peer-lifecycle callbacks
        # registered on that same engine during SyftRDSClient.__init__.
        managers = tuple(c._rds.sync_engine for c in clients)
        setup_callbacks(managers)
        write_versions(managers)
        wire_peers(managers)

        return clients
