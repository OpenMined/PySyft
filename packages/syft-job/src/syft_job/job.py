from __future__ import annotations

import shutil
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from .read_permissions import (
    LOG_FILES,
    ReadPermissions,
    check_approval_reason,
    validate_read_permissions,
    format_read_permissions,
    restrict_to_request,
    warn_on_logs_release,
)
from .job_repr import (
    StderrViewer,
    job_info_repr_html,
    jobs_list_repr_html,
    jobs_list_str,
)
from .job_stdout import StdoutViewer
from .job_storage import JobRef
from .models import JobState, JobStatus, JobSubmissionMetadata
from .policies import (
    OUTPUTS_WITHHELD_FILENAME,
    POLICY_ERROR_FILENAME,
    JobPolicies,
    PolicyResolutionError,
)
from .submission import API_HEADER, OwnerJobRecord, submission_hash

if TYPE_CHECKING:
    from .client import JobClient


class JobInfo:
    """Represents a job with data from both inbox/ and review/ directories."""

    def __init__(
        self,
        job_metadata: JobSubmissionMetadata,
        state: JobState,
        current_user_email: str,
        client: JobClient,
        ref: JobRef,
    ):
        self.job_metadata = job_metadata
        self._state = state
        self.current_user_email = current_user_email
        self._client = client
        # Identity (owner, submitter, name) comes from the path-derived ref,
        # never from the DS-writable config.yaml.
        self._ref = ref
        self.job_headers = dict(job_metadata.headers)

    @property
    def datasite_owner_email(self) -> str:
        return self._ref.datasite_email

    @property
    def job_submission_path(self) -> Path:
        return self._client.manager.submission_dir(self._ref)

    @property
    def job_review_path(self) -> Path:
        return self._client.manager.review_dir(self._ref)

    def share_outputs(self, recipients: list[str]) -> None:
        """Share the outputs with ``recipients``."""
        self.share({"outputs": recipients})

    def share_logs(self, recipients: list[str]) -> None:
        """Share the logs and the exit code with ``recipients``.

        This call cannot take back a copy that another party already has.
        """
        self.share({name: recipients for name in LOG_FILES})

    def share(self, files: Mapping[str, list[str]]) -> None:
        """Grant each recipient read access to its named file in review/.

        ``files`` maps each file name to its recipients. A name with no
        recipients, or with no file, is skipped. A user with read access on the
        whole review folder reads every file, shared or not.
        """
        present = {
            name: recipients
            for name, recipients in files.items()
            if recipients and (self.job_review_path / name).exists()
        }
        if present:
            self._grant_read(present)

    # ──────────────────────────────────────────────
    # Read permissions
    # ──────────────────────────────────────────────

    @property
    def requested_read_permissions(self) -> ReadPermissions:
        """Who the submitter asked to read which file, as recorded at receipt."""
        return self._state.requested_read_permissions

    @property
    def approved_read_permissions(self) -> ReadPermissions:
        """Who the data owner lets read which file, as recorded at approval."""
        return self._state.approved_read_permissions

    def _set_approved_read_permissions(
        self, read_permissions: Optional[ReadPermissions], stacklevel: int = 4
    ) -> ReadPermissions:
        """Keep the part of ``read_permissions`` that the submitter requested."""
        approved = restrict_to_request(
            validate_read_permissions(read_permissions), self.requested_read_permissions
        )
        # Up to the public caller, update_read_permissions() by default.
        warn_on_logs_release(approved, stacklevel)
        self._state.approved_read_permissions = approved
        return approved

    def read_permission_rows(self) -> list[tuple[str, str]]:
        """The read permission state as (label, value) rows, for display."""
        return [
            ("Requested", format_read_permissions(self.requested_read_permissions)),
            ("Approved", format_read_permissions(self.approved_read_permissions)),
        ]

    def update_read_permissions(
        self, read_permissions: Optional[ReadPermissions]
    ) -> ReadPermissions:
        """Replace who may read which file, and return what was approved.

        Only what the submitter requested is kept. A job that finished shares
        the new files at once. A read already given is not taken back.

        Raises:
            ValueError: If the job was not approved
            PermissionError: If the current user is not the datasite owner
        """
        if self.datasite_owner_email != self.current_user_email:
            raise PermissionError(
                f"Only the admin user ({self.datasite_owner_email}) can change "
                f"the read permissions of a job."
            )
        self._state = self._client.manager.read_state(self._ref)
        if self._state.status in (
            JobStatus.RECEIVED,
            JobStatus.PENDING,
            JobStatus.REJECTED,
        ):
            raise ValueError(f"Approve the job first (current: {self.status}).")
        approved = self._set_approved_read_permissions(read_permissions)
        self._client.manager.write_state(self._ref, self._state)
        if self._state.status in (JobStatus.DONE, JobStatus.FAILED):
            self.share_read_permissions()
        return approved

    def share_read_permissions(self) -> None:
        """Give each approved reader read access to its file."""
        self.share(self.approved_read_permissions)

    # ──────────────────────────────────────────────
    # Properties from config (inbox/)
    # ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._ref.job_name

    @property
    def submitted_by(self) -> str:
        return self._ref.ds_email

    @property
    def submitted_at(self) -> Optional[str]:
        if self.job_metadata.submitted_at:
            return self.job_metadata.submitted_at.isoformat()
        return None

    @property
    def code_dir(self) -> Path:
        """Path to the submitted code directory."""
        return self.job_submission_path / "code"

    @property
    def code(self) -> str:
        """Read the entrypoint source code."""
        ep = self.job_metadata.entrypoint
        if ep:
            ep_path = self.code_dir / ep
            if ep_path.exists():
                return ep_path.read_text()
        # fallback: first .py file in code/
        for f in self.code_dir.rglob("*.py"):
            return f.read_text()
        return ""

    @property
    def run_script(self) -> str:
        """Read the run.sh content."""
        run_sh = self.job_submission_path / "run.sh"
        if run_sh.exists():
            return run_sh.read_text()
        return ""

    # ──────────────────────────────────────────────
    # Properties from state (review/)
    # ──────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._state.status.value

    @property
    def output_paths(self) -> List[Path]:
        """Get list of all file paths in the outputs directory."""
        status = self._state.status

        def _list_output_files() -> List[Path]:
            outputs_dir = self.job_review_path / "outputs"
            if not outputs_dir.exists():
                return []
            try:
                return [
                    item
                    for item in outputs_dir.iterdir()
                    if item.name != PERMISSION_FILE_NAME
                ]
            except OSError:
                return []

        if status == JobStatus.FAILED:
            partial = _list_output_files()
            print(f"❌ Job '{self.name}' failed (status: {self.status}).")
            print(
                f"   Returning {len(partial)} partial output file(s). "
                "Check stderr for details: job.stderr"
            )
            return partial

        if status != JobStatus.DONE:
            print(f"⏳ Job '{self.name}' is not done yet (status: {self.status}).")
            print("   Sync and check again: client.sync()")
            return []

        output_files = _list_output_files()
        if not output_files:
            print(f"⚠️  Job '{self.name}' is done but no output files were found.")
            print("   The job script may not have written to the outputs/ folder.")
            return []

        return output_files

    @property
    def stdout(self) -> StdoutViewer:
        """Get a viewer for the stdout content."""
        return StdoutViewer(self)

    @property
    def stderr(self) -> StderrViewer:
        """Get a viewer for the stderr content."""
        return StderrViewer(self)

    def _log_path(self, log_file: str) -> Path:
        """Where the viewers read ``log_file`` ("stdout.txt" or "stderr.txt")."""
        return self.job_review_path / log_file

    @property
    def files(self) -> List[Path]:
        """Get list of relevant files across both inbox and review."""
        all_files = []
        try:
            for root in (self.job_submission_path, self.job_review_path):
                if not root.exists():
                    continue
                for f in root.rglob("*"):
                    if not f.is_file():
                        continue
                    if f.name.startswith("."):
                        continue
                    if any(
                        d in f.parts
                        for d in (".venv", "__pycache__", ".git", "node_modules")
                    ):
                        continue
                    all_files.append(f)
        except OSError:
            pass
        return all_files

    # ──────────────────────────────────────────────
    # Actions (write to review/)
    # ──────────────────────────────────────────────

    @property
    def approval_method(self) -> Optional[str]:
        return self._state.approval_method

    @property
    def review_reason(self) -> Optional[str]:
        """Reason recorded at review time (approval or rejection)."""
        return self._state.review_reason

    @property
    def api_name(self) -> Optional[str]:
        """The api this job calls.

        On the DO's side it comes from the owner record, which only the DO
        writes. Elsewhere it is the name the caller put in config.yaml.
        """
        record = self._client.manager.read_owner_record(self._ref)
        if record is not None:
            return record.api_name
        return self.job_headers.get(API_HEADER)

    @property
    def policies(self) -> Optional[JobPolicies]:
        """The policies of the api this job calls, when this client loads apis."""
        resolver = self._client.api_resolver
        if self.api_name is None or resolver is None:
            return None
        try:
            return resolver(self.api_name).policies
        except PolicyResolutionError:
            return None

    @property
    def policy_error(self) -> Optional[str]:
        """Why a policy denied the job, as the submitter sees it."""
        path = self.job_review_path / POLICY_ERROR_FILENAME
        try:
            return path.read_text().strip() or None
        except OSError:
            return None

    @property
    def outputs_withheld(self) -> bool:
        """True when the output policy kept the outputs from the submitter.

        The runner shares an empty marker file with the submitter for this;
        only whether it exists matters.
        """
        return (self.job_review_path / OUTPUTS_WITHHELD_FILENAME).exists()

    def approve(
        self,
        reason: Optional[str] = None,
        approval_method: str = "manual",
        share_files: Optional[ReadPermissions] = None,
        share_requested_files: bool = False,
    ) -> None:
        """
        Approve a job by updating state.yaml in review/.
        Only the datasite owner can approve jobs.

        The approval covers the submission as it was received, and records its
        hash. The runner executes only a copy that matches it, so a later edit
        of the submitter's inbox never runs.

        Args:
            reason: Optional reason for approval (recorded as review_reason).
            approval_method: How the job was approved ("manual" or "auto")
            share_files: Who may read which file once the job ran, as
                {file: [emails]}, e.g. {"stdout.txt": [submitter]}. Only what the
                submitter requested is kept.
            share_requested_files: Instead of share_files, approve
                everything the submitter requested. With neither, none is approved.

        Raises:
            ValueError: If job is not in pending status, was not fully
                received, or changed after it was received, or if both
                share_files and share_requested_files are given
            PermissionError: If the current user is not authorized to approve
            TypeError: If reason is not a string, or share_files not a dict
        """
        check_approval_reason(reason)
        resolved_read_permissions = self._resolve_read_permissions(
            share_files, share_requested_files
        )
        self._approve(reason, approval_method, resolved_read_permissions)

    def _resolve_read_permissions(
        self, share_files: object, share_requested_files: bool
    ) -> ReadPermissions:
        """The approval's read permissions: the whole request, or share_files."""
        if share_files is not None and share_requested_files:
            raise ValueError("Give share_files or share_requested_files, not both.")
        if share_requested_files:
            return dict(self.requested_read_permissions)
        return validate_read_permissions(share_files)

    def _approve(
        self,
        reason: Optional[str],
        approval_method: str,
        read_permissions: Optional[ReadPermissions],
    ) -> None:
        self._check_can_approve()
        # The check runs before any write. The record goes before the state, so
        # the runner never sees an approved state without it.
        record = self._read_owner_record()
        self._check_unchanged(record)
        self._record_approval(record, approval_method)
        # Up to the caller of approve(), through _approve().
        self._set_approved_read_permissions(read_permissions, stacklevel=5)
        self._state.status = JobStatus.APPROVED
        self._state.approved_by = self.current_user_email
        self._state.approved_at = record.approved_at
        self._state.approval_method = approval_method
        self._state.review_reason = reason
        self._client.manager.write_state(self._ref, self._state)
        print(f"✅ Job '{self.name}' approved successfully!")
        print("   Status    : approved → will run on next process cycle")
        print("\n⏳ Next step: run process_approved_jobs() to execute it.")
        print("   client.process_approved_jobs(share_outputs_with_submitter=True)")

    def _check_can_approve(self) -> None:
        if self._state.status != JobStatus.PENDING:
            raise ValueError(
                f"Job '{self.name}' is not in pending status (current: {self.status})"
            )
        if self.datasite_owner_email != self.current_user_email:
            raise PermissionError(
                f"You are {self.current_user_email}, and job '{self.name}' is on "
                f"{self.datasite_owner_email}'s datasite. Only they can approve "
                f"it. If you meant one of your own, select your datasite "
                f'first: jobs["{self.current_user_email}"]["<name>"].'
            )

    def record_approval(self, approval_method: str) -> None:
        """Record the received submission as approved, without changing the state.

        For an owner whose approval is decided elsewhere, such as an enclave,
        which runs a job once every data owner approved this same submission.
        The runner executes a job only with this record.

        Raises:
            ValueError: If the job was not fully received, or changed after.
        """
        record = self._read_owner_record()
        self._check_unchanged(record)
        self._record_approval(record, approval_method)

    def _read_owner_record(self) -> OwnerJobRecord:
        """The owner-side record of the job, written when it was received."""
        record = self._client.manager.read_owner_record(self._ref)
        if record is None:
            raise ValueError(
                f"Job '{self.name}' was not fully received yet. List the jobs "
                f"again once all its files arrived, then approve."
            )
        return record

    def _check_unchanged(self, record: OwnerJobRecord) -> None:
        """Raise unless the submission is the one received.

        The runner checks its run copy against the approved hash anyway; this
        gives a clear error at approval instead of a failed job later.
        """
        if submission_hash(self.job_submission_path) != record.received_hash:
            raise ValueError(
                f"Job '{self.name}' changed after it was received, so it is not "
                f"approved. Reject it, or ask the submitter to submit it again."
            )

    def _record_approval(self, record: OwnerJobRecord, approval_method: str) -> None:
        """Record the received submission as approved, and how, in one write."""
        record.approved_hash = record.received_hash
        record.approved_by = self.current_user_email
        record.approved_at = datetime.now(timezone.utc)
        record.approval_method = approval_method
        self._client.manager.write_owner_record(self._ref, record)

    def reject(self, reason: Optional[str] = None) -> None:
        """
        Reject a job by updating state.yaml in review/.

        Args:
            reason: Optional reason for rejection (recorded as review_reason).

        Raises:
            ValueError: If job is not in pending status
            PermissionError: If the current user is not authorized to reject
        """
        if self._state.status != JobStatus.PENDING:
            raise ValueError(
                f"Job '{self.name}' is not in pending status (current: {self.status})"
            )

        if self.datasite_owner_email != self.current_user_email:
            raise PermissionError(
                f"You are {self.current_user_email}, and job '{self.name}' is on "
                f"{self.datasite_owner_email}'s datasite. Only they can reject it."
            )

        self._state.status = JobStatus.REJECTED
        self._state.rejected_by = self.current_user_email
        self._state.rejected_at = datetime.now(timezone.utc)
        self._state.review_reason = reason
        self._client.manager.write_state(self._ref, self._state)
        print(f"Job '{self.name}' rejected.")

    def accept_by_depositing_result(self, path: str) -> Path:
        """
        Accept a job by depositing the result file or folder and marking as done.

        Args:
            path: Path to the result file or folder to deposit

        Returns:
            Path to the deposited result in the review/outputs directory

        Raises:
            ValueError: If job is not in pending or approved status
            FileNotFoundError: If the result file or folder doesn't exist
        """
        if self._state.status not in (JobStatus.PENDING, JobStatus.APPROVED):
            raise ValueError(
                f"Job '{self.name}' is not in pending or approved status (current: {self.status})"
            )

        result_path = Path(path)
        if not result_path.exists():
            raise FileNotFoundError(f"Result path not found: {path}")

        outputs_dir = self.job_review_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Handle both files and folders
        result_name = result_path.name
        destination = outputs_dir / result_name

        if result_path.is_file():
            shutil.copy2(str(result_path), str(destination))
        elif result_path.is_dir():
            shutil.copytree(str(result_path), str(destination))
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

        # Update state
        now = datetime.now(timezone.utc)
        self._state.status = JobStatus.DONE
        self._state.approved_by = self._state.approved_by or self.current_user_email
        self._state.approved_at = self._state.approved_at or now
        self._state.completed_at = now
        self._state.return_code = 0
        self._client.manager.write_state(self._ref, self._state)

        print(
            f"Job '{self.name}' completed successfully! Result deposited at: {destination}"
        )

        return destination

    def rerun(self) -> None:
        """
        Rerun a job by cleaning up review/ artifacts and resetting to approved.

        The runner executes a fresh copy that must still match the approved
        hash. A job that was never approved, for example one accepted by
        depositing a result, is sent back to pending by the runner instead.

        Raises:
            ValueError: If job is not in done or failed status
        """
        if self._state.status not in (JobStatus.DONE, JobStatus.FAILED):
            raise ValueError(
                f"Job '{self.name}' is not in done/failed status (current: {self.status}). "
                f"Only completed or failed jobs can be rerun."
            )

        removed = self._client.manager.reset_run(self._ref)

        # Reset state to approved
        self._state.status = JobStatus.APPROVED
        self._state.completed_at = None
        self._state.return_code = None
        self._client.manager.write_state(self._ref, self._state)

        if removed:
            print(
                f"Job '{self.name}' prepared for rerun! Removed: {', '.join(removed)}"
            )
        else:
            print(f"Job '{self.name}' prepared for rerun! (No cleanup needed)")

    # ──────────────────────────────────────────────
    # Permissions (review/)
    # ──────────────────────────────────────────────

    def _get_perm_context(self):
        from syft_perms import SyftPermContext

        datasite = self._client.config.syftbox_folder / self.datasite_owner_email
        return SyftPermContext(datasite=datasite)

    def _relative_review_path(self, subpath: str) -> Path:
        """Return path relative to the datasite for a review/ subpath."""
        rel = self.job_review_path.relative_to(
            self._client.config.syftbox_folder / self.datasite_owner_email
        )
        return rel / subpath

    def _grant_read(self, files: Mapping[str, list[str]]) -> None:
        """Grant each recipient read access to its named review/ file."""
        ctx = self._get_perm_context()
        for filename, recipients in files.items():
            f = ctx.open(self._relative_review_path(filename))
            for recipient in recipients:
                f.grant_read_access(recipient)

    # ──────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        status_emojis = {
            "received": "📨",
            "pending": "📥",
            "approved": "✅",
            "rejected": "❌",
            "running": "🔄",
            "done": "🎉",
            "failed": "💥",
        }
        emoji = status_emojis.get(self.status, "❓")
        approval_info = ""
        if self.approval_method:
            approval_info = f" [approved: {self.approval_method}]"
        base = f"{emoji} {self.name} ({self.status}{approval_info}) -> {self.datasite_owner_email}"
        if self.review_reason:
            base += f" | Review reason: {self.review_reason}"
        rows = "; ".join(
            f"{label}: {value}" for label, value in self.read_permission_rows()
        )
        return f"{base} | Read permissions: {rows}"

    def __repr__(self) -> str:
        parts = [
            f"name='{self.name}'",
            f"submitted_by='{self.submitted_by}'",
            f"current_user_email='{self.current_user_email}'",
            f"status='{self.status}'",
        ]
        if self.approval_method:
            parts.append(f"approval_method='{self.approval_method}'")
        if self.review_reason:
            parts.append(f"review_reason='{self.review_reason}'")
        return f"JobInfo({', '.join(parts)})"

    def _repr_html_(self) -> str:
        return job_info_repr_html(self)


def _with_party(jobs: List[JobInfo], email: str) -> List[JobInfo]:
    """The jobs ``email`` is a party to: on its datasite, or submitted by it."""
    return [j for j in jobs if email in (j.datasite_owner_email, j.submitted_by)]


def _with_name(jobs: List[JobInfo], name: str) -> List[JobInfo]:
    """The jobs called ``name``."""
    return [j for j in jobs if j.name == name]


def _with_submission(
    jobs: List[JobInfo], datasite: str, submitter: str
) -> List[JobInfo]:
    """The jobs on ``datasite`` submitted by ``submitter``."""
    return [
        j
        for j in jobs
        if j.datasite_owner_email == datasite and j.submitted_by == submitter
    ]


def _keep(jobs: List[JobInfo], key: str, first_email: str | None) -> List[JobInfo]:
    """The jobs one subscript key keeps, read the way ``__getitem__`` reads it.

    ``first_email`` is the email an earlier key in the chain gave, or None.
    The '@' tells the two kinds of key apart, so the usage hint can measure a
    chain before offering it. A deprecated job name holding an '@' reads here
    as an email and keeps nothing, where ``__getitem__`` falls back to the
    name; the hint then offers a position, which is the safe direction to err.
    """
    if "@" not in key:
        return _with_name(jobs, key)
    if first_email is None:
        return _with_party(jobs, key)
    return _with_submission(jobs, first_email, key)


class JobsList:
    """A list-like container for JobInfo objects with nice display."""

    def __init__(
        self,
        jobs: List[JobInfo],
        root_email: str,
        has_do_role: bool = False,
        first_email: str | None = None,
    ):
        self._jobs = jobs
        self._root_email = root_email
        self._has_do_role = has_do_role
        # The email that selected this list, which a second email key reads as
        # the datasite. None for a list no email has selected.
        self._first_email = first_email

    def __getitem__(self, index: int | str) -> "JobInfo | JobsList":
        """A job by position or name, or the jobs of one party by email.

        A first email keeps the jobs it is a party to, on either side. That is
        usually the other party — a data scientist names the data owner, a data
        owner names the submitter — but naming yourself keeps your own. So
        ``jobs["do@x.org"]["analysis"]`` reads as one job. A second email names
        the submitter, and the first one is then the datasite:
        ``jobs["do@x.org"]["ds@y.org"]["analysis"]`` pins the pair a job name is
        unique under, even when the two people sent each other a job of that
        name. A bare name searches
        every datasite at once and raises when more than one job answers to it.
        Job names cannot contain ``@``, so the two kinds of key never collide.
        """
        if isinstance(index, int):
            return self._jobs[index]
        elif isinstance(index, str):
            if "@" in index:
                return self._by_email(index)
            return self._by_name(index)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _by_email(self, email: str) -> "JobInfo | JobsList":
        """The jobs this email is a party to, or, after a first email, submitted.

        A first email covers both roles because which one narrows depends on
        who is asking. A data scientist names the data owner's datasite; a data
        owner names the submitter. A second email has to fix the roles, because
        two people with both roles can each hold a job of one name that the
        other submitted, and each email is a party to both.

        A job submitted before names could not hold an '@' answers to no party,
        and reading it is the one thing this key must not take away, so a key
        that names no party falls back to the name.
        """
        matches = _keep(self._jobs, email, self._first_email)
        if matches:
            return JobsList(
                matches,
                self._root_email,
                self._has_do_role,
                first_email=self._first_email or email,
            )

        if any(job.name == email for job in self._jobs):
            warnings.warn(
                f"Job name {email!r} holds an '@', which now marks a datasite or "
                "submitter email. Such names are deprecated and the next version "
                "will not resolve them. Rename the job.",
                DeprecationWarning,
                stacklevel=3,
            )
            return self._by_name(email)

        datasites = ", ".join(sorted({job.datasite_owner_email for job in self._jobs}))
        submitters = ", ".join(sorted({job.submitted_by for job in self._jobs}))
        raise ValueError(
            f"No jobs involving {email}. These jobs are on: {datasites}. "
            f"They were submitted by: {submitters}."
        )

    def _by_name(self, name: str) -> JobInfo:
        matches = _with_name(self._jobs, name)
        if not matches:
            raise ValueError(f"Job with name '{name}' not found")
        if len(matches) > 1:
            raise ValueError(self._ambiguous_name_message(name, matches))
        return matches[0]

    def _ambiguous_name_message(self, name: str, matches: List[JobInfo]) -> str:
        """Why the name did not resolve, and a subscript chain for each job.

        An email key keeps jobs on either side, so one email can be the
        datasite of one candidate and the submitter of another. Each chain is
        measured with ``_reaches_only`` before it is offered. One submitter
        holding a name twice across protocol layouts defeats every chain, so
        such a job gets its position in this list.
        """
        lines = [
            f"  {self._accessor_for(job)} — on {job.datasite_owner_email} "
            f"from {job.submitted_by}"
            for job in matches
        ]
        return (
            f"Multiple jobs are named '{name}'. Select one by subscripting this "
            "list with:\n" + "\n".join(lines)
        )

    def _accessor_for(self, job: JobInfo) -> str:
        """The shortest subscript chain that reaches ``job`` alone, or its position."""
        chains = (
            (job.datasite_owner_email, job.name),
            (job.submitted_by, job.name),
            (job.datasite_owner_email, job.submitted_by, job.name),
        )
        for keys in chains:
            if self._reaches_only(keys, job):
                return "".join(f'["{key}"]' for key in keys)
        return f"[{self._jobs.index(job)}]"

    def hint_accessor(self) -> str | None:
        """The subscript chain the jobs table tells a data owner to type.

        Names a pending job on the DO's own datasite and gives the shortest
        chain that reaches it and nothing else. The email in a chain is the
        other party, so the submitter comes first; the datasite joins it only
        when that submitter used the name on another datasite too. One
        submitter holding a name twice across protocol layouts defeats every
        chain, which is what the position is for.

        Returns None when no such job is there to name, because the hint's
        ``approve()`` takes only a pending job on your own datasite: a client
        with no data-owner role, a DO who owns none of these jobs, or a DO
        whose own jobs have all been reviewed already. The hint also offers
        ``accept_by_depositing_result()``, which an approved job would still
        take; withholding both is the cost of never printing a command that
        raises.
        """
        if not self._has_do_role:
            return None
        owned = [
            j
            for j in self._jobs
            if j.datasite_owner_email == self._root_email and j.status == "pending"
        ]
        if not owned:
            return None
        pick = owned[0]
        chains = (
            (pick.submitted_by, pick.name),
            (self._root_email, pick.submitted_by, pick.name),
        )
        for keys in chains:
            if self._reaches_only(keys, pick):
                return "".join(f'["{key}"]' for key in keys)
        return f"[{self._jobs.index(pick)}]"

    def _reaches_only(self, keys: tuple[str, ...], job: JobInfo) -> bool:
        """Whether subscripting by ``keys`` in turn reaches ``job`` and nothing else."""
        reached, first_email = self._jobs, self._first_email
        for key in keys:
            reached = _keep(reached, key, first_email)
            if "@" in key and first_email is None:
                first_email = key
        return len(reached) == 1 and reached[0] is job

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self):
        return iter(self._jobs)

    def __str__(self) -> str:
        return jobs_list_str(self._jobs, self.hint_accessor())

    def __repr__(self) -> str:
        return f"JobsList({len(self._jobs)} jobs)"

    def _repr_html_(self) -> str:
        return jobs_list_repr_html(self._jobs, self.hint_accessor())
