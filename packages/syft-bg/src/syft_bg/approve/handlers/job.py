"""Job approval handler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from syft_job.job import JobInfo

from syft_bg.approve.config import (
    AutoApprovalObj,
    AutoApprovalsConfig,
    AutoApproveConfig,
)
from syft_bg.approve.criteria import (
    AutoApprovalValidationResult,
    _validate_job_against_object,
)

if TYPE_CHECKING:
    from syft_client.sync.syftbox_manager import SyftboxManager


class StateManager(Protocol):
    """Protocol for state management."""

    def mark_approved(self, job_name: str, submitted_by: str) -> None: ...
    def was_approved(self, job_name: str) -> bool: ...


class JobApprovalHandler:
    """Handles job approval logic."""

    def __init__(
        self,
        client: SyftboxManager,
        config_path: Optional[Path] = None,
        state: Optional[StateManager] = None,
        on_approve: Optional[Callable[[JobInfo], None]] = None,
        verbose: bool = True,
    ):
        self.client = client
        self._config_path = config_path
        self.state = state
        self.on_approve = on_approve
        self.verbose = verbose
        self._skipped_jobs: set[str] = set()

    @property
    def config(self) -> AutoApprovalsConfig:
        """Always-fresh auto-approvals config, re-read from disk on every access."""
        return AutoApproveConfig.load(self._config_path).auto_approvals

    def _get_approved_peers(self) -> list[str]:
        """Get list of approved peer emails."""
        self.client.load_peers(force_download=True)
        return [p.email for p in self.client.peer_manager.approved_peers]

    def evaluate_auto_approval(self, job: JobInfo) -> AutoApprovalValidationResult:
        """Find matching auto-approval objects for a job and validate.

        Searches all objects where the peer is listed (or peers is empty = any peer).
        Any matching object wins.
        """
        if job.status != "pending":
            return AutoApprovalValidationResult(
                match=False, reason=f"status is {job.status}, not pending"
            )

        candidate_objects: list[tuple[str, AutoApprovalObj]] = []
        for name, obj in self.config.objects.items():
            if not obj.peers or job.submitted_by in obj.peers:
                candidate_objects.append((name, obj))

        if not candidate_objects:
            return AutoApprovalValidationResult(
                match=False,
                reason=f"no auto-approval objects match peer: {job.submitted_by}",
            )

        last_reason = ""
        for name, obj in candidate_objects:
            result = _validate_job_against_object(job, obj)
            if result.match:
                return AutoApprovalValidationResult(match=True, reason="ok")
            last_reason = f"[{name}] {result.reason}"

        return AutoApprovalValidationResult(match=False, reason=last_reason)

    def check_and_approve(self) -> list[JobInfo]:
        """Check all jobs and approve those matching criteria."""
        if not self.config.enabled:
            return []

        approved_jobs = []

        for job in self.client.jobs:
            if self.state and self.state.was_approved(job.name):
                continue

            result = self.evaluate_auto_approval(job)

            if not result.match:
                if job.status == "pending" and job.name not in self._skipped_jobs:
                    self._skipped_jobs.add(job.name)
                    if self.verbose:
                        print(f"Skipped: {job.name} ({result.reason})")
                continue

            try:
                job.approve(approval_method="auto")
                approved_jobs.append(job)

                if self.state:
                    self.state.mark_approved(job.name, job.submitted_by)

                if self.on_approve:
                    self.on_approve(job)

                if self.verbose:
                    print(f"Approved: {job.name} from {job.submitted_by}")

            except Exception as e:
                if self.verbose:
                    print(f"Failed to approve {job.name}: {e}")

        if approved_jobs:
            self.client.process_approved_jobs(
                stream_output=self.verbose,
                share_outputs_with_submitter=True,
                share_logs_with_submitter=True,
            )

        return approved_jobs
