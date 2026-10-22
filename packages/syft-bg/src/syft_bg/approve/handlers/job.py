"""Job approval handler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from syft_job.job import JobInfo
from syft_rds.apis.matching import check_request, matching_apis

from syft_bg.approve.api_store import ApiStore
from syft_bg.approve.config import AutoApprovalsConfig
from syft_bg.approve.criteria import AutoApprovalValidationResult
from syft_bg.common.syft_bg_config import SyftBgConfig

if TYPE_CHECKING:
    from syft_rds import SyftRDSClient


class StateManager(Protocol):
    """Protocol for state management."""

    def mark_approved(self, job_name: str, submitted_by: str) -> None: ...
    def was_approved(self, job_name: str) -> bool: ...


class JobApprovalHandler:
    """Handles job approval logic."""

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
        self._skipped_jobs: set[str] = set()

    @property
    def config(self) -> AutoApprovalsConfig:
        """Always-fresh auto-approvals config, re-read from disk on every access."""
        return SyftBgConfig.load(self._config_path).approve.auto_approvals

    @property
    def api_store(self) -> ApiStore:
        return ApiStore(self.client.syftbox_folder, self.client.email)

    @property
    def default_disclosures(self) -> list[str]:
        """Re-read from disk on every access, like ``config``."""
        return SyftBgConfig.load(self._config_path).approve.default_disclosures

    def _get_approved_peers(self) -> list[str]:
        """Get list of approved peer emails."""
        self.client.load_peers(force_download=True)
        return [p.email for p in self.client.peer_manager.approved_peers]

    def evaluate_auto_approval(self, job: JobInfo) -> AutoApprovalValidationResult:
        """Whether a job may be approved automatically, and through which api.

        A request is checked against the api it names. A whole job must match
        exactly one api, among those shared with its submitter.
        """
        if job.status != "pending":
            return AutoApprovalValidationResult(
                match=False, reason=f"status is {job.status}, not pending"
            )
        if job.is_request:
            return self._evaluate_request(job)
        return self._evaluate_whole_job(job)

    def _evaluate_request(self, job: JobInfo) -> AutoApprovalValidationResult:
        # The name comes from the submitter, so it must be one of the apis.
        if job.api_name not in self.api_store.names():
            return AutoApprovalValidationResult(
                match=False, reason=f"calls no api of yours: {job.api_name!r}"
            )
        definition = self.api_store.get(job.api_name)
        ok, reason = check_request(
            job.job_submission_path, definition, job.submitted_by
        )
        return AutoApprovalValidationResult(
            match=ok, reason=reason, api_name=job.api_name if ok else None
        )

    def _evaluate_whole_job(self, job: JobInfo) -> AutoApprovalValidationResult:
        """A whole job never runs under policies, so it can't match an api with
        policies: that api is called through a request instead."""
        matches, last_reason = matching_apis(
            self.api_store.apis_dir, job.job_submission_path, job.submitted_by
        )
        if len(matches) > 1:
            return AutoApprovalValidationResult(
                match=False, reason=f"matches several apis: {matches}"
            )
        if not matches:
            return AutoApprovalValidationResult(match=False, reason=last_reason)
        obj = self.api_store.get(matches[0])
        if obj.input_policy or obj.output_policy:
            return AutoApprovalValidationResult(
                match=False,
                reason=f"api '{matches[0]}' has policies; call it through client.api",
            )
        return AutoApprovalValidationResult(
            match=True, reason="ok", api_name=matches[0]
        )

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
                job.approve(
                    approval_method="auto", disclosures=self.default_disclosures
                )
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
            )

        return approved_jobs
