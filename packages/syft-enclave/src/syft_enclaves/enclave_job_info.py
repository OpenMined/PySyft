from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, Field, field_validator
from syft_job.read_permissions import (
    ReadPermissions,
    check_approval_reason,
    check_one_way,
    check_read_permissions,
    format_read_permissions,
    restrict_to_request,
    valid_read_permissions,
    warn_on_logs_release,
)
from syft_job.job import JobInfo
from syft_job.models import JobStatus
from syft_job.submission import submission_hash  # noqa: F401 (re-exported)


class PartyApprovalStatus(BaseModel):
    """Tracks approval from a single party in a multi-party (enclave) job."""

    party: str
    dataset: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    approved_at: Optional[datetime] = None
    # Who this party lets read which file, as {file: [emails]}. Absent means
    # nobody, so an approval file from an older client approves nothing.
    approved_read_permissions: dict[str, list[str]] = Field(default_factory=dict)
    # What the party approved: submission_hash() of the job it reviewed. An
    # approval counts only while the job still hashes to this value.
    submission_hash: Optional[str] = None
    rejected_at: Optional[datetime] = None
    reason: Optional[str] = None

    @field_validator("approved_read_permissions", mode="before")
    @classmethod
    def _drop_invalid_read_permissions(cls, value: object) -> ReadPermissions:
        """Keep only the valid entries, so an item-based older file approves nothing."""
        return valid_read_permissions(value)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f)

    @classmethod
    def load_json(cls, path: Path) -> PartyApprovalStatus:
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def enclave_approval_file_name(do_email: str) -> str:
    return f"{do_email}_approval_state.json"


def load_enclave_approval_files(review_dir: Path) -> list[PartyApprovalStatus]:
    """Load all *_approval_state.json files from review_dir."""
    if not review_dir.exists():
        return []
    results = []
    for f in sorted(review_dir.glob("*_approval_state.json")):
        results.append(PartyApprovalStatus.load_json(f))
    return results


def load_party_approval(review_dir: Path, party: str) -> PartyApprovalStatus | None:
    """The approval file of ``party``, or None when missing, unreadable or another party's."""
    try:
        approval = PartyApprovalStatus.load_json(
            review_dir / enclave_approval_file_name(party)
        )
    except (OSError, ValueError):
        return None
    if approval.party.casefold() != party.casefold():
        return None
    return approval


class EnclaveJobInfo(JobInfo):
    """Enclave-specific JobInfo with multi-party approval logic.

    ``_required_approvers`` is set only on the enclave, which runs the job:
    there every configured data owner must approve this exact submission. A
    data owner's client holds only its own approval file, so it keeps showing
    the status of the files it has.
    """

    _required_approvers: Optional[list[str]] = None

    # Called with the path of the approval file after each change, so the
    # enclave client can push the file. None pushes nothing.
    _on_approval_change: Optional[Callable[[Path], None]] = None

    @classmethod
    def from_job_info(
        cls,
        job: JobInfo,
        required_approvers: Optional[list[str]] = None,
        on_approval_change: Optional[Callable[[Path], None]] = None,
    ) -> EnclaveJobInfo:
        instance = cls.__new__(cls)
        instance.__dict__.update(job.__dict__)
        instance._required_approvers = required_approvers
        if on_approval_change is not None:
            instance._on_approval_change = on_approval_change
        return instance

    @property
    def _approval_file(self) -> Path:
        return self.job_review_path / enclave_approval_file_name(
            self.current_user_email
        )

    def _load_own_approval(self) -> PartyApprovalStatus:
        if not self._approval_file.exists():
            raise PermissionError(
                f"No approval file for {self.current_user_email} on job "
                f"'{self.name}'. The enclave writes one per designated party "
                f"when it distributes the job, so either it has not distributed "
                f"this job yet — run client.sync() and retry — or you are not a "
                f"party to it."
            )
        return PartyApprovalStatus.load_json(self._approval_file)

    def _save_own_approval(self, approval: PartyApprovalStatus) -> None:
        approval.save_json(self._approval_file)
        if self._on_approval_change is not None:
            self._on_approval_change(self._approval_file)

    @property
    def approved_read_permissions(self) -> ReadPermissions:
        """A party's own approval, from its approval file.

        Anyone without an approval file, such as the enclave or the submitter,
        reads what every party approved, as the enclave recorded it in the state.
        """
        if not self._approval_file.exists():
            return super().approved_read_permissions
        return PartyApprovalStatus.load_json(
            self._approval_file
        ).approved_read_permissions

    @property
    def read_permissions_to_share(self) -> ReadPermissions:
        """Per file, the emails that the submitter requested and every party approved.

        Raises:
            LookupError: Off the enclave. A data owner holds only its own
                approval file, so it reads its own in ``approved_read_permissions``.
        """
        if self.current_user_email != self.datasite_owner_email:
            raise LookupError(
                "Only the enclave holds the approval of every party. "
                "Read this party's approval in `approved_read_permissions`."
            )
        return read_permissions_approved_by_all_parties(
            self.job_review_path,
            self.requested_read_permissions,
            self._required_approvers,
        )

    @property
    def status(self) -> str:
        if self._state.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.RUNNING):
            return self._state.status.value
        if self._required_approvers is not None:
            return self._status_from_required_approvers(self._required_approvers)
        approvals = load_enclave_approval_files(self.job_review_path)
        if not approvals:
            return self._state.status.value
        if any(a.status == JobStatus.REJECTED for a in approvals):
            return JobStatus.REJECTED.value
        if all(a.status == JobStatus.APPROVED for a in approvals):
            return JobStatus.APPROVED.value
        return JobStatus.PENDING.value

    @property
    def can_approve(self) -> bool:
        """Whether the enclave has sent us our approval file yet.

        A submitter sees its own job as soon as it submits, before the enclave
        has distributed it, so seeing the job is not enough to approve it.
        """
        return self._approval_file.exists()

    def _status_from_required_approvers(self, required: list[str]) -> str:
        """Approved only when every required party approved this exact submission.

        A missing or unreadable approval file counts as no approval, so removing
        a file never stands in for consent. A rejection from any party wins.
        """
        if not required:
            return JobStatus.PENDING.value
        approvals = [load_party_approval(self.job_review_path, p) for p in required]
        if any(a is not None and a.status == JobStatus.REJECTED for a in approvals):
            return JobStatus.REJECTED.value
        current = submission_hash(self.job_submission_path)
        if all(_approves(a, current) for a in approvals):
            return JobStatus.APPROVED.value
        return JobStatus.PENDING.value

    def read_permission_rows(self) -> list[tuple[str, str]]:
        """The read permission state as (label, value) rows, for display.

        The enclave shows the approval of every party and what all approved.
        A data owner shows its own approval.
        """
        rows = [("Requested", format_read_permissions(self.requested_read_permissions))]
        if self.current_user_email == self.datasite_owner_email:
            return rows + self._party_rows()
        if self._approval_file.exists():
            own = format_read_permissions(self.approved_read_permissions)
            rows.append(("Approved by you", own))
            rows.append(("Approved by all", "needs the approval of every party"))
        return rows

    def _party_rows(self) -> list[tuple[str, str]]:
        """Each party's approval, then what every party approved."""
        approvals = _party_approvals(self.job_review_path, self._required_approvers)
        rows = [(f"Approved by {p}", _approval_summary(a)) for p, a in approvals]
        shared = format_read_permissions(self.read_permissions_to_share)
        return [*rows, ("Approved by all", shared)]

    def _restrict_to_request(
        self, read_permissions: Optional[ReadPermissions]
    ) -> ReadPermissions:
        """The part of ``read_permissions`` the submitter requested.

        Raises:
            TypeError, ValueError: If ``read_permissions`` is not {file: [emails]}.
        """
        checked = check_read_permissions(read_permissions)
        return restrict_to_request(checked, self.requested_read_permissions)

    def _read_permissions_to_approve(
        self, read_permissions: object, share_requested_files: Optional[bool]
    ) -> ReadPermissions:
        """The whole request, or the requested part of ``read_permissions``."""
        if share_requested_files is None:
            share_requested_files = read_permissions is None
        flags = {"share_requested_files": share_requested_files}
        check_one_way("read_permissions", read_permissions, flags)
        if share_requested_files:
            return dict(self.requested_read_permissions)
        return self._restrict_to_request(read_permissions)

    def approve(
        self,
        reason: Optional[str] = None,
        approval_method: str = "manual",
        read_permissions: Optional[ReadPermissions] = None,
        share_requested_files: Optional[bool] = None,
    ) -> None:
        """Approve this submission in the DO's own approval file.

        ``read_permissions`` says who may read which file, as {file: [emails]}.
        ``share_requested_files`` None means True unless read_permissions is
        given: with neither, the approval covers everything the submitter requested.
        The enclave shares a file with an email only when every party approved
        it, and only what the submitter requested is kept. The file records the
        hash of the submission, so the enclave runs the job only while it still
        matches. ``reason`` and ``approval_method`` are not recorded.

        Raises:
            ValueError: If read_permissions comes with share_requested_files=True.
        """
        check_approval_reason(reason)
        approved = self._read_permissions_to_approve(
            read_permissions, share_requested_files
        )
        approval = self._load_own_approval()
        current = submission_hash(self.job_submission_path)
        if approval.status == JobStatus.REJECTED or _approves(approval, current):
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.APPROVED
        approval.approved_at = datetime.now(timezone.utc)
        approval.submission_hash = current
        approval.approved_read_permissions = approved
        warn_on_logs_release(approved)
        self._save_own_approval(approval)
        print(f"Job '{self.name}' approved by {self.current_user_email}!")

    def update_read_permissions(
        self, read_permissions: Optional[ReadPermissions]
    ) -> ReadPermissions:
        """Replace who this party lets read which file, and return what was kept.

        The enclave applies the change on its next cycle.
        """
        approved = self._restrict_to_request(read_permissions)
        approval = self._load_own_approval()
        if approval.status != JobStatus.APPROVED:
            raise ValueError(
                f"Approve the job first (current: {approval.status.value})."
            )
        approval.approved_read_permissions = approved
        warn_on_logs_release(approved)
        self._save_own_approval(approval)
        return approved

    def reject(self, reason: Optional[str] = None) -> None:
        """Reject this job in the DO's own approval file.

        The inherited ``JobInfo.reject`` writes ``state.yaml``, which only the
        enclave owns. A rejection here is a written refusal the enclave reads,
        and it also withdraws an earlier approval.
        """
        approval = self._load_own_approval()
        if approval.status == JobStatus.REJECTED:
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.REJECTED
        approval.rejected_at = datetime.now(timezone.utc)
        approval.reason = reason
        self._save_own_approval(approval)
        print(f"Job '{self.name}' rejected by {self.current_user_email}.")


def _approval_summary(approval: Optional[PartyApprovalStatus]) -> str:
    """One party's approval for display, or why it has none."""
    if approval is None:
        return "(no approval file)"
    if approval.status != JobStatus.APPROVED:
        return f"({approval.status.value})"
    return format_read_permissions(approval.approved_read_permissions)


def _approves(approval: PartyApprovalStatus | None, current_hash: str) -> bool:
    return (
        approval is not None
        and approval.status == JobStatus.APPROVED
        and approval.submission_hash == current_hash
    )


def submission_approvals(
    review_dir: Path, current_hash: str, parties: Iterable[str]
) -> list[PartyApprovalStatus]:
    """The approvals from ``parties`` of the submission with ``current_hash``.

    An approval of another version of the submission is left out, as it is
    when the enclave decides whether to run the job.
    """
    return [
        a
        for _, a in _party_approvals(review_dir, parties)
        if _approves(a, current_hash)
    ]


def _party_approvals(
    review_dir: Path, parties: Optional[Iterable[str]]
) -> list[tuple[str, Optional[PartyApprovalStatus]]]:
    """Each party with its approval, or None when its file is missing or invalid.

    With ``parties`` None, the parties are the approval files present.
    """
    if parties is None:
        return [(a.party, a) for a in load_enclave_approval_files(review_dir)]
    return [(p, load_party_approval(review_dir, p)) for p in parties]


def read_permissions_approved_by_all_parties(
    review_dir: Path,
    requested: ReadPermissions,
    parties: Optional[Iterable[str]] = None,
) -> ReadPermissions:
    """Per file, the emails in ``requested`` that every party approved.

    Empty when a party has not approved, or a named party has no valid approval
    file. With ``parties`` None, the parties are the approval files present.
    """
    approvals = [a for _, a in _party_approvals(review_dir, parties)]
    if not approvals or not all(
        a is not None and a.status == JobStatus.APPROVED for a in approvals
    ):
        return {}
    approved = requested
    for approval in approvals:
        approved = restrict_to_request(approval.approved_read_permissions, approved)
    return approved
