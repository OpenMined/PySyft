from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, Field
from syft_job.disclosures import (  # noqa: F401 (re-exported)
    DISCLOSURE_ITEMS,
    DisclosureItem,
    DisclosuresArg,
    check_approval_reason,
    format_items,
    normalize_disclosures,
    restrict_to_request,
    warn_on_logs_release,
)
from syft_job.job import JobInfo
from syft_job.models import JobStatus


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

    # Called with the path of the approval file after each change, so the
    # enclave client can push the file. None pushes nothing.
    _on_approval_change: Optional[Callable[[Path], None]] = None

    @classmethod
    def from_job_info(
        cls,
        job: JobInfo,
        on_approval_change: Optional[Callable[[Path], None]] = None,
    ) -> EnclaveJobInfo:
        instance = cls.__new__(cls)
        instance.__dict__.update(job.__dict__)
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
                f"No approval file found for {self.current_user_email}. "
                f"You may not be a designated party for this job."
            )
        return PartyApprovalStatus.load_json(self._approval_file)

    def _save_own_approval(self, approval: PartyApprovalStatus) -> None:
        # Up to the public caller: approve() or update_disclosures().
        warn_on_logs_release(
            approval.disclosures, self.requested_disclosures, stacklevel=4
        )
        approval.save_json(self._approval_file)
        if self._on_approval_change is not None:
            self._on_approval_change(self._approval_file)

    @property
    def disclosures(self) -> dict[str, bool]:
        """The items this party released, as recorded in its approval file."""
        if not self._approval_file.exists():
            return {}
        return normalize_disclosures(
            PartyApprovalStatus.load_json(self._approval_file).disclosures
        )

    @property
    def granted_disclosures(self) -> set[str]:
        """The items that go out: requested, and released by every party.

        Raises:
            LookupError: Off the enclave. A data owner holds only its own
                approval file, so it reads its own grant in ``disclosures``.
        """
        if self.current_user_email != self.datasite_owner_email:
            raise LookupError(
                "Only the enclave holds the grant of every party. "
                "Read this party's grant in `disclosures`."
            )
        return approved_disclosures(self.job_review_path, self.requested_disclosures)

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

    def disclosure_rows(self) -> list[tuple[str, str]]:
        """The disclosure state as (label, value) rows, for display.

        The enclave holds the approval file of every party, so it shows each
        grant and the combined result. A data owner shows its own grant.
        """
        rows = [("Requested", format_items(self.requested_disclosures))]
        if self.current_user_email == self.datasite_owner_email:
            for approval in load_enclave_approval_files(self.job_review_path):
                if approval.status == JobStatus.APPROVED:
                    value = format_items(normalize_disclosures(approval.disclosures))
                else:
                    value = f"({approval.status.value})"
                rows.append((f"Granted by {approval.party}", value))
            rows.append(("To submitter", format_items(self.granted_disclosures)))
        elif self._approval_file.exists():
            rows.append(("Your grant", format_items(self.disclosures)))
            rows.append(("To submitter", "needs the grant of every party"))
        return rows

    def approve(
        self,
        reason: Optional[str] = None,
        approval_method: str = "manual",
        disclosures: DisclosuresArg = None,
    ) -> None:
        """Write approval to the DO's individual approval state file.

        ``disclosures`` names the items in ``DisclosureItem`` that this party
        releases to the other parties. The enclave releases an item only when
        every party released it. Omit the argument to release nothing.

        The approval file does not record ``reason`` or ``approval_method``.
        Only the items that the submitter requested are stored, so a later
        edit of the request cannot add items.
        """
        check_approval_reason(reason)
        approval = self._load_own_approval()
        if approval.status != JobStatus.PENDING:
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.APPROVED
        approval.approved_at = datetime.now(timezone.utc)
        approval.disclosures = restrict_to_request(
            disclosures, self.requested_disclosures
        )
        self._save_own_approval(approval)
        print(f"Job '{self.name}' approved by {self.current_user_email}!")

    def update_disclosures(self, disclosures: DisclosuresArg) -> dict[str, bool]:
        """Replace the items this party releases, and return the new map.

        The enclave applies the new set on its next cycle.
        """
        approval = self._load_own_approval()
        if approval.status != JobStatus.APPROVED:
            raise ValueError(
                f"Approve the job first (current: {approval.status.value})."
            )
        approval.disclosures = restrict_to_request(
            disclosures, self.requested_disclosures
        )
        self._save_own_approval(approval)
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
