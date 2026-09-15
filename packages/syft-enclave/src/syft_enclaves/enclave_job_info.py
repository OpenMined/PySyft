from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union

from pydantic import BaseModel, Field
from syft_job.job import JobInfo
from syft_job.models import JobStatus


class DisclosureItem(str, Enum):
    """A class of job data that a party can release to the other parties.

    - ``TRACEBACK_FRAMES``: the failure position in the approved code, and the
      builtin exception type. Bounded.
    - ``LOGS``: stdout and stderr. Unbounded, and the job chooses every byte.
    """

    TRACEBACK_FRAMES = "traceback_frames"
    LOGS = "logs"


DISCLOSURE_ITEMS = frozenset(item.value for item in DisclosureItem)


def normalize_disclosures(
    items: Union[str, DisclosureItem, Iterable[str], None],
) -> dict[str, bool]:
    """Return the known items in ``items`` as a map. Unknown names are dropped.

    A single name is accepted on its own, because iterating a string would
    produce its characters and grant nothing.
    """
    if not items:
        return {}
    if isinstance(items, (str, DisclosureItem)):
        items = [items]
    names = {str(getattr(i, "value", i)) for i in items}
    return {name: True for name in sorted(names & DISCLOSURE_ITEMS)}


class PartyApprovalStatus(BaseModel):
    """Tracks approval from a single party in a multi-party (enclave) job."""

    party: str
    dataset: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    approved_at: Optional[datetime] = None
    # The items this party releases. Absent means the party released nothing,
    # so an approval file from an older client grants nothing.
    disclosures: dict[str, bool] = Field(default_factory=dict)

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

    def approve(self, disclosures: Optional[Iterable[str]] = None) -> None:
        """Write approval to the DO's individual approval state file.

        ``disclosures`` names the items in ``DisclosureItem`` that this party
        releases to the other parties. The enclave releases an item only when
        every party released it. Omit the argument to release nothing.
        """
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
        approval.disclosures = normalize_disclosures(disclosures)
        approval.save_json(approval_file)
        print(f"Job '{self.name}' approved by {self.current_user_email}!")

    def update_disclosures(self, disclosures: Optional[Iterable[str]]) -> dict:
        """Replace the items this party releases, and return the new map.

        A copy that already reached another party does not come back.
        """
        file_name = enclave_approval_file_name(self.current_user_email)
        approval_file = self.job_review_path / file_name
        if not approval_file.exists():
            raise PermissionError(
                f"No approval file found for {self.current_user_email}. "
                f"You may not be a designated party for this job."
            )
        approval = PartyApprovalStatus.load_json(approval_file)
        if approval.status != JobStatus.APPROVED:
            raise ValueError(
                f"Approve the job first (current: {approval.status.value})."
            )
        approval.disclosures = normalize_disclosures(disclosures)
        approval.save_json(approval_file)
        return approval.disclosures


def approved_disclosures(
    review_dir: Path, requested: Optional[Iterable[str]] = None
) -> set[str]:
    """Return the items that every party released.

    An item needs the agreement of all parties, therefore the result is the
    intersection of the parties' grants. The result is empty if a party did not
    approve the job yet, because a party grants nothing before it approves.
    When ``requested`` is given, the result holds only the requested items.
    """
    approvals = load_enclave_approval_files(review_dir)
    if not approvals:
        return set()

    granted: Optional[set[str]] = None
    for approval in approvals:
        if approval.status != JobStatus.APPROVED:
            return set()
        party_grant = {
            name
            for name, allowed in approval.disclosures.items()
            if allowed and name in DISCLOSURE_ITEMS
        }
        granted = party_grant if granted is None else granted & party_grant

    result = granted or set()
    if requested is not None:
        result = result & set(normalize_disclosures(requested))
    return result
