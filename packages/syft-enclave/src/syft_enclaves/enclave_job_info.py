from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from syft_job.job import JobInfo
from syft_job.models.state import JobStatus


class PartyApprovalStatus(BaseModel):
    """Tracks approval from a single party in a multi-party (enclave) job."""

    party: str
    dataset: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    approved_at: Optional[datetime] = None

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


class EnclaveJobInfo(JobInfo):
    """Enclave-specific JobInfo with multi-party approval logic."""

    @classmethod
    def from_job_info(cls, job: JobInfo) -> EnclaveJobInfo:
        instance = cls.__new__(cls)
        instance.__dict__.update(job.__dict__)
        return instance

    @property
    def status(self) -> str:
        if self._state.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.RUNNING):
            return self._state.status.value
        approvals = load_enclave_approval_files(self.job_review_path)
        if not approvals:
            return self._state.status.value
        if any(a.status == JobStatus.REJECTED for a in approvals):
            return JobStatus.REJECTED.value
        if all(a.status == JobStatus.APPROVED for a in approvals):
            return JobStatus.APPROVED.value
        return JobStatus.PENDING.value

    def approve(self) -> None:
        """Write approval to the DO's individual approval state file."""
        file_name = enclave_approval_file_name(self.current_user_email)
        approval_file = self.job_review_path / file_name
        if not approval_file.exists():
            raise PermissionError(
                f"No approval file found for {self.current_user_email}. "
                f"You may not be a designated party for this job."
            )
        approval = PartyApprovalStatus.load_json(approval_file)
        if approval.status != JobStatus.PENDING:
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.APPROVED
        approval.approved_at = datetime.now(timezone.utc)
        approval.save_json(approval_file)
        print(f"Job '{self.name}' approved by {self.current_user_email}!")
