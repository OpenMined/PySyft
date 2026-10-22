"""Runs the jobs that api calls became."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from syft_job.job import JobInfo

from syft_bg.approve.config import AutoApprovalsConfig
from syft_bg.common.syft_bg_config import SyftBgConfig

if TYPE_CHECKING:
    from syft_rds import SyftRDSClient


class StateManager(Protocol):
    """Protocol for state management."""

    def mark_approved(self, job_name: str, submitted_by: str) -> None: ...
    def was_approved(self, job_name: str) -> bool: ...


class ApprovedJobHandler:
    """Runs the approved jobs on the DO's datasite.

    An api call arrives approved: the DO's client turns the request into an
    approved job when it lists jobs. A job the DO approved by hand runs too.
    """

    def __init__(
        self,
        client: SyftRDSClient,
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

    @property
    def config(self) -> AutoApprovalsConfig:
        """Always-fresh config, re-read from disk on every access."""
        return SyftBgConfig.load(self._config_path).approve.auto_approvals

    def run_approved_jobs(self) -> list[JobInfo]:
        """Run every approved job; return those not seen approved before."""
        if not self.config.enabled:
            return []
        approved = [
            job
            for job in self.client.jobs
            if job.status == "approved"
            and job.datasite_owner_email == self.client.email
        ]
        new = self._record_new(approved)
        if approved:
            self.client.process_approved_jobs(
                stream_output=self.verbose,
                share_outputs_with_submitter=True,
            )
        return new

    def _record_new(self, approved: list[JobInfo]) -> list[JobInfo]:
        """Mark the jobs not seen before, and call on_approve for each."""
        new = [
            job
            for job in approved
            if not (self.state and self.state.was_approved(job.name))
        ]
        for job in new:
            if self.state:
                self.state.mark_approved(job.name, job.submitted_by)
            if self.on_approve:
                self.on_approve(job)
            if self.verbose:
                print(f"Running: {job.name} from {job.submitted_by}")
        return new
