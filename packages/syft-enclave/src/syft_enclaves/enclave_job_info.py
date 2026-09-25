from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from syft_job.job import JobInfo
from syft_job.models import JobStatus
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME


class PartyApprovalStatus(BaseModel):
    """Tracks approval from a single party in a multi-party (enclave) job."""

    party: str
    dataset: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    approved_at: Optional[datetime] = None
    # What the party approved: submission_hash() of the job it reviewed. An
    # approval counts only while the job still hashes to this value.
    submission_hash: Optional[str] = None
    rejected_at: Optional[datetime] = None
    reason: Optional[str] = None

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


def submission_hash(submission_dir: Path) -> str:
    """SHA-256 over every file of a job submission: code/, run.sh and config.yaml.

    config.yaml carries the job name, the submission time and the datasets, so
    one digest pins the job, its code and its dataset set. Permission files are
    left out: the enclave writes them after the data owners got their copy.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in submission_dir.rglob("*") if p.is_file()):
        if path.name == PERMISSION_FILE_NAME:
            continue
        digest.update(path.relative_to(submission_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class EnclaveJobInfo(JobInfo):
    """Enclave-specific JobInfo with multi-party approval logic.

    ``_required_approvers`` is set only on the enclave, which runs the job:
    there every configured data owner must approve this exact submission. A
    data owner's client holds only its own approval file, so it keeps showing
    the status of the files it has.
    """

    _required_approvers: Optional[list[str]] = None

    @classmethod
    def from_job_info(
        cls, job: JobInfo, required_approvers: Optional[list[str]] = None
    ) -> EnclaveJobInfo:
        instance = cls.__new__(cls)
        instance.__dict__.update(job.__dict__)
        instance._required_approvers = required_approvers
        return instance

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

    def _own_approval_file(self) -> Path:
        approval_file = self.job_review_path / enclave_approval_file_name(
            self.current_user_email
        )
        if not approval_file.exists():
            raise PermissionError(
                f"No approval file found for {self.current_user_email}. "
                f"You may not be a designated party for this job."
            )
        return approval_file

    def approve(self) -> None:
        """Approve this submission in the DO's own approval file.

        The file records the hash of the submission the DO has, so the enclave
        runs the job only while it still matches. A DO may approve again after
        the submission changed.
        """
        approval_file = self._own_approval_file()
        approval = PartyApprovalStatus.load_json(approval_file)
        current = submission_hash(self.job_submission_path)
        if approval.status == JobStatus.REJECTED or _approves(approval, current):
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.APPROVED
        approval.approved_at = datetime.now(timezone.utc)
        approval.submission_hash = current
        approval.save_json(approval_file)
        print(f"Job '{self.name}' approved by {self.current_user_email}!")

    def reject(self, reason: Optional[str] = None) -> None:
        """Reject this job in the DO's own approval file.

        The inherited ``JobInfo.reject`` writes ``state.yaml``, which only the
        enclave owns. A rejection here is a written refusal the enclave reads,
        and it also withdraws an earlier approval.
        """
        approval_file = self._own_approval_file()
        approval = PartyApprovalStatus.load_json(approval_file)
        if approval.status == JobStatus.REJECTED:
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.REJECTED
        approval.rejected_at = datetime.now(timezone.utc)
        approval.reason = reason
        approval.save_json(approval_file)
        print(f"Job '{self.name}' rejected by {self.current_user_email}.")


def _approves(approval: PartyApprovalStatus | None, current_hash: str) -> bool:
    return (
        approval is not None
        and approval.status == JobStatus.APPROVED
        and approval.submission_hash == current_hash
    )
