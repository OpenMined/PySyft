from __future__ import annotations

import hashlib
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
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME


class PartyApprovalStatus(BaseModel):
    """Tracks approval from a single party in a multi-party (enclave) job."""

    party: str
    dataset: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    approved_at: Optional[datetime] = None
    # The items this party releases. Absent means the party released nothing,
    # so an approval file from an older client grants nothing.
    disclosures: dict[str, bool] = Field(default_factory=dict)
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
                f"No approval file found for {self.current_user_email}. "
                f"You may not be a designated party for this job."
            )
        return PartyApprovalStatus.load_json(self._approval_file)

    def _save_own_approval(self, approval: PartyApprovalStatus) -> None:
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
        return approved_disclosures(
            self.job_review_path, self.requested_disclosures, self._required_approvers
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

    def disclosure_rows(self) -> list[tuple[str, str]]:
        """The disclosure state as (label, value) rows, for display.

        The enclave holds the approval file of every party, so it shows each
        grant and the combined result. A data owner shows its own grant.
        """
        rows = [("Requested", format_items(self.requested_disclosures))]
        if self.current_user_email == self.datasite_owner_email:
            for party, approval in _party_approvals(
                self.job_review_path, self._required_approvers
            ):
                if approval is None:
                    value = "(no approval file)"
                elif approval.status == JobStatus.APPROVED:
                    value = format_items(normalize_disclosures(approval.disclosures))
                else:
                    value = f"({approval.status.value})"
                rows.append((f"Granted by {party}", value))
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
        """Approve this submission in the DO's own approval file.

        ``disclosures`` names the items in ``DisclosureItem`` that this party
        releases to the other parties. The enclave releases an item only when
        every party released it. Omit the argument to release nothing. Only the
        items that the submitter requested are stored, so a later edit of the
        request cannot add items.

        The file records the hash of the submission the DO has, so the enclave
        runs the job only while it still matches. A DO may approve again after
        the submission changed. The approval file does not record ``reason`` or
        ``approval_method``.
        """
        check_approval_reason(reason)
        approval = self._load_own_approval()
        current = submission_hash(self.job_submission_path)
        if approval.status == JobStatus.REJECTED or _approves(approval, current):
            raise ValueError(f"Already in status: {approval.status.value}")
        approval.status = JobStatus.APPROVED
        approval.approved_at = datetime.now(timezone.utc)
        approval.submission_hash = current
        approval.disclosures = restrict_to_request(
            disclosures, self.requested_disclosures
        )
        warn_on_logs_release(approval.disclosures, self.requested_disclosures)
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
        warn_on_logs_release(approval.disclosures, self.requested_disclosures)
        self._save_own_approval(approval)
        return approval.disclosures

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


def _approves(approval: PartyApprovalStatus | None, current_hash: str) -> bool:
    return (
        approval is not None
        and approval.status == JobStatus.APPROVED
        and approval.submission_hash == current_hash
    )


def _party_approvals(
    review_dir: Path, parties: Optional[Iterable[str]]
) -> list[tuple[str, Optional[PartyApprovalStatus]]]:
    """Each party with its approval, or None when its file is missing or invalid.

    With ``parties`` None, the parties are the approval files present.
    """
    if parties is None:
        return [(a.party, a) for a in load_enclave_approval_files(review_dir)]
    return [(p, load_party_approval(review_dir, p)) for p in parties]


def approved_disclosures(
    review_dir: Path,
    requested: Optional[Iterable[str]] = None,
    parties: Optional[Iterable[str]] = None,
) -> set[str]:
    """Return the items that every party released.

    An item needs the agreement of all parties, therefore the result is the
    intersection of the parties' grants. The result is empty if a party did not
    approve the job yet, because a party grants nothing before it approves.
    When ``requested`` is given, the result holds only the requested items.

    ``parties`` names the parties that must grant. A named party without a valid
    approval file grants nothing. With ``parties`` None, the parties are the
    approval files present.
    """
    approvals = _party_approvals(review_dir, parties)
    if not approvals:
        return set()

    granted: Optional[set[str]] = None
    for _, approval in approvals:
        if approval is None or approval.status != JobStatus.APPROVED:
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
