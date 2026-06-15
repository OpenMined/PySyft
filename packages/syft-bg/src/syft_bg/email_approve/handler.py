"""Handler for email-based job approval/rejection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from syft_bg.common.state import JsonStateManager

if TYPE_CHECKING:
    from syft_job.client import JobClient
    from syft_job.job_runner import SyftJobRunner


class EmailAction(Enum):
    APPROVE = "approve"
    AUTO_APPROVE = "auto_approve"
    DENY = "deny"
    UNKNOWN = "unknown"


@dataclass
class EmailApprovalResponse:
    action: EmailAction
    reason: Optional[str] = None


def parse_reply(text: str) -> EmailApprovalResponse:
    """Parse a reply email for approve/deny commands."""
    if not text:
        return EmailApprovalResponse(action=EmailAction.UNKNOWN)

    # Take first non-empty line
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith(("auto-approve", "auto approve", "autoapprove")):
            return EmailApprovalResponse(action=EmailAction.AUTO_APPROVE)

        if lower.startswith("approve") or lower.startswith("aprove"):
            return EmailApprovalResponse(action=EmailAction.APPROVE)

        match = re.match(r"deny\s+(.*)", line, re.IGNORECASE)
        if match:
            reason = match.group(1).strip()
            return EmailApprovalResponse(
                action=EmailAction.DENY,
                reason=reason if reason else "No reason provided",
            )

        if lower == "deny":
            return EmailApprovalResponse(
                action=EmailAction.DENY, reason="No reason provided"
            )

        # First non-empty line didn't match any command
        return EmailApprovalResponse(action=EmailAction.UNKNOWN)

    return EmailApprovalResponse(action=EmailAction.UNKNOWN)


class EmailApproveHandler:
    """Processes email replies to approve or reject jobs."""

    def __init__(
        self,
        job_client: JobClient,
        job_runner: SyftJobRunner,
        state: JsonStateManager,
        notify_state: JsonStateManager,
        do_email: str,
    ):
        self.job_client = job_client
        self.job_runner = job_runner
        self.state = state
        self.notify_state = notify_state
        self.do_email = do_email

    def handle_reply(self, thread_id: str, reply_text: str) -> None:
        """Process a reply email for a job thread."""
        job_name = self.notify_state.get_job_name_by_thread_id(thread_id)
        if not job_name:
            raise ValueError(f"No job found for thread {thread_id}")
        else:
            print(f"[EmailApproveHandler] Job found for thread {thread_id}: {job_name}")

        state_key = f"email_reply_{thread_id}"
        if self.state.was_notified(state_key, "processed"):
            return

        response = parse_reply(reply_text)
        if response.action == EmailAction.UNKNOWN:
            raise ValueError(f"Unrecognized reply for {job_name}: {reply_text[:100]}")

        job = self._find_job(job_name)

        if job.status != "pending":
            raise ValueError(f"Job {job_name} is {job.status}, not pending")

        if response.action == EmailAction.APPROVE:
            self._approve_job(job, job_name, state_key)
        elif response.action == EmailAction.AUTO_APPROVE:
            self._auto_approve_job(job, job_name, state_key)
        elif response.action == EmailAction.DENY:
            self._reject_job(job, job_name, response.reason or "", state_key)

    def _find_job(self, job_name: str):
        """Find a job by name in the client's job list."""
        for job in self.job_client.jobs:
            if job.name == job_name:
                return job
        raise ValueError(f"Job not found: {job_name}")

    def _approve_job(
        self, job, job_name: str, state_key: str, approval_method: str = "manual"
    ) -> None:
        """Approve a job, execute it, and share results."""
        job.approve(approval_method=approval_method)
        self.state.mark_notified(state_key, "processed")

        self.job_runner.process_approved_jobs(
            share_outputs_with_submitter=True,
            share_logs_with_submitter=True,
        )
        print(f"[EmailApproveHandler] Approved job: {job_name}")

    def _auto_approve_job(self, job, job_name: str, state_key: str) -> None:
        """Approve a job and create an auto-approval object for future jobs."""
        self._approve_job(job, job_name, state_key, approval_method="auto")

        from syft_bg.api.api import auto_approve_job

        result = auto_approve_job(job)
        if result.success:
            print(f"[EmailApproveHandler] Auto-approval created for: {job_name}")
        else:
            print(
                f"[EmailApproveHandler] Failed to create auto-approval for {job_name}: {result.error}"
            )

    def _reject_job(self, job, job_name: str, reason: str, state_key: str) -> None:
        """Reject a job with a reason."""
        job.reject(reason)
        self.state.mark_notified(state_key, "processed")
        print(f"[EmailApproveHandler] Rejected job: {job_name} (reason: {reason})")
