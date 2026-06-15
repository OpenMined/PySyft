from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from .job_repr import (
    StderrViewer,
    job_info_repr_html,
    jobs_list_repr_html,
    jobs_list_str,
)
from .job_stdout import StdoutViewer
from .models.config import JobSubmissionMetadata
from .models.state import JobState, JobStatus

if TYPE_CHECKING:
    from .client import JobClient


class JobInfo:
    """Represents a job with data from both inbox/ and review/ directories."""

    def __init__(
        self,
        job_metadata: JobSubmissionMetadata,
        state: JobState,
        datasite_owner_email: str,
        current_user_email: str,
        client: JobClient,
    ):
        self.job_metadata = job_metadata
        self._state = state
        self.datasite_owner_email = datasite_owner_email
        self.current_user_email = current_user_email
        self._client = client
        self.job_headers = dict(job_metadata.headers)

    @property
    def job_submission_path(self) -> Path:
        return self._client.config.get_job_submission_dir(
            self.job_metadata.datasite_email, self.job_metadata.submitted_by, self.name
        )

    @property
    def job_review_path(self) -> Path:
        return self._client.config.get_review_job_dir(
            self.job_metadata.datasite_email, self.job_metadata.submitted_by, self.name
        )

    # ──────────────────────────────────────────────
    # Properties from config (inbox/)
    # ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.job_metadata.name

    @property
    def submitted_by(self) -> str:
        return self.job_metadata.submitted_by

    @property
    def submitted_at(self) -> Optional[str]:
        if self.job_metadata.submitted_at:
            return self.job_metadata.submitted_at.isoformat()
        return None

    @property
    def code_dir(self) -> Path:
        """Path to the submitted code directory."""
        return self.job_submission_path / "code"

    @property
    def code(self) -> str:
        """Read the entrypoint source code."""
        ep = self.job_metadata.entrypoint
        if ep:
            ep_path = self.code_dir / ep
            if ep_path.exists():
                return ep_path.read_text()
        # fallback: first .py file in code/
        for f in self.code_dir.rglob("*.py"):
            return f.read_text()
        return ""

    @property
    def run_script(self) -> str:
        """Read the run.sh content."""
        run_sh = self.job_submission_path / "run.sh"
        if run_sh.exists():
            return run_sh.read_text()
        return ""

    # ──────────────────────────────────────────────
    # Properties from state (review/)
    # ──────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._state.status.value

    @property
    def output_paths(self) -> List[Path]:
        """Get list of all file paths in the outputs directory."""
        status = self._state.status

        def _list_output_files() -> List[Path]:
            outputs_dir = self.job_review_path / "outputs"
            if not outputs_dir.exists():
                return []
            try:
                return [
                    item
                    for item in outputs_dir.iterdir()
                    if item.name != PERMISSION_FILE_NAME
                ]
            except Exception:
                return []

        if status == JobStatus.FAILED:
            partial = _list_output_files()
            print(f"❌ Job '{self.name}' failed (status: {self.status}).")
            print(
                f"   Returning {len(partial)} partial output file(s). "
                "Check stderr for details: job.stderr"
            )
            return partial

        if status != JobStatus.DONE:
            print(f"⏳ Job '{self.name}' is not done yet (status: {self.status}).")
            print("   Sync and check again: client.sync()")
            return []

        output_files = _list_output_files()
        if not output_files:
            print(f"⚠️  Job '{self.name}' is done but no output files were found.")
            print("   The job script may not have written to the outputs/ folder.")
            return []

        return output_files

    @property
    def stdout(self) -> StdoutViewer:
        """Get a viewer for the stdout content."""
        return StdoutViewer(self)

    @property
    def stderr(self) -> StderrViewer:
        """Get a viewer for the stderr content."""
        return StderrViewer(self)

    @property
    def files(self) -> List[Path]:
        """Get list of relevant files across both inbox and review."""
        all_files = []
        try:
            for root in (self.job_submission_path, self.job_review_path):
                if not root.exists():
                    continue
                for f in root.rglob("*"):
                    if not f.is_file():
                        continue
                    if f.name.startswith("."):
                        continue
                    if any(
                        d in f.parts
                        for d in (".venv", "__pycache__", ".git", "node_modules")
                    ):
                        continue
                    all_files.append(f)
        except Exception:
            pass
        return all_files

    # ──────────────────────────────────────────────
    # Actions (write to review/)
    # ──────────────────────────────────────────────

    @property
    def approval_method(self) -> Optional[str]:
        return self._state.approval_method

    @property
    def review_reason(self) -> Optional[str]:
        """Reason recorded at review time (approval or rejection)."""
        return self._state.review_reason

    def approve(
        self, reason: Optional[str] = None, approval_method: str = "manual"
    ) -> None:
        """
        Approve a job by updating state.yaml in review/.
        Only the datasite owner can approve jobs.

        Args:
            reason: Optional reason for approval (recorded as review_reason).
            approval_method: How the job was approved ("manual" or "auto")

        Raises:
            ValueError: If job is not in pending status
            PermissionError: If the current user is not authorized to approve
        """
        if self._state.status != JobStatus.PENDING:
            raise ValueError(
                f"Job '{self.name}' is not in pending status (current: {self.status})"
            )

        if self.datasite_owner_email != self.current_user_email:
            raise PermissionError(
                f"Only the admin user ({self.datasite_owner_email}) can approve jobs in their folder. "
                f"Current job is in {self.datasite_owner_email}'s folder."
            )

        self._state.status = JobStatus.APPROVED
        self._state.approved_by = self.current_user_email
        self._state.approved_at = datetime.now(timezone.utc)
        self._state.approval_method = approval_method
        self._state.review_reason = reason
        self._state.save(self.job_review_path / "state.yaml")
        print(f"✅ Job '{self.name}' approved successfully!")
        print("   Status    : approved → will run on next process cycle")
        print("\n⏳ Next step: run process_approved_jobs() to execute it.")
        print("   client.process_approved_jobs(share_outputs_with_submitter=True)")

    def reject(self, reason: Optional[str] = None) -> None:
        """
        Reject a job by updating state.yaml in review/.

        Args:
            reason: Optional reason for rejection (recorded as review_reason).

        Raises:
            ValueError: If job is not in pending status
            PermissionError: If the current user is not authorized to reject
        """
        if self._state.status != JobStatus.PENDING:
            raise ValueError(
                f"Job '{self.name}' is not in pending status (current: {self.status})"
            )

        if self.datasite_owner_email != self.current_user_email:
            raise PermissionError(
                f"Only the admin user ({self.datasite_owner_email}) can reject jobs."
            )

        self._state.status = JobStatus.REJECTED
        self._state.rejected_by = self.current_user_email
        self._state.rejected_at = datetime.now(timezone.utc)
        self._state.review_reason = reason
        self._state.save(self.job_review_path / "state.yaml")
        print(f"Job '{self.name}' rejected.")

    def accept_by_depositing_result(self, path: str) -> Path:
        """
        Accept a job by depositing the result file or folder and marking as done.

        Args:
            path: Path to the result file or folder to deposit

        Returns:
            Path to the deposited result in the review/outputs directory

        Raises:
            ValueError: If job is not in pending or approved status
            FileNotFoundError: If the result file or folder doesn't exist
        """
        if self._state.status not in (JobStatus.PENDING, JobStatus.APPROVED):
            raise ValueError(
                f"Job '{self.name}' is not in pending or approved status (current: {self.status})"
            )

        result_path = Path(path)
        if not result_path.exists():
            raise FileNotFoundError(f"Result path not found: {path}")

        # Create outputs directory in review/
        outputs_dir = self.job_review_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Handle both files and folders
        result_name = result_path.name
        destination = outputs_dir / result_name

        if result_path.is_file():
            shutil.copy2(str(result_path), str(destination))
        elif result_path.is_dir():
            shutil.copytree(str(result_path), str(destination))
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

        # Update state
        now = datetime.now(timezone.utc)
        self._state.status = JobStatus.DONE
        self._state.approved_by = self._state.approved_by or self.current_user_email
        self._state.approved_at = self._state.approved_at or now
        self._state.completed_at = now
        self._state.return_code = 0
        self._state.save(self.job_review_path / "state.yaml")

        print(
            f"Job '{self.name}' completed successfully! Result deposited at: {destination}"
        )

        return destination

    def rerun(self) -> None:
        """
        Rerun a job by cleaning up review/ artifacts and resetting to approved.

        Raises:
            ValueError: If job is not in done or failed status
        """
        if self._state.status not in (JobStatus.DONE, JobStatus.FAILED):
            raise ValueError(
                f"Job '{self.name}' is not in done/failed status (current: {self.status}). "
                f"Only completed or failed jobs can be rerun."
            )

        changes_made = []

        # Clean up review/ artifacts
        for filename in ("stdout.txt", "stderr.txt", "returncode.txt"):
            f = self.job_review_path / filename
            if f.exists():
                f.unlink()
                changes_made.append(filename)

        outputs_dir = self.job_review_path / "outputs"
        if outputs_dir.exists() and outputs_dir.is_dir():
            shutil.rmtree(outputs_dir)
            changes_made.append("outputs directory")

        # Reset state to approved
        self._state.status = JobStatus.APPROVED
        self._state.completed_at = None
        self._state.return_code = None
        self._state.save(self.job_review_path / "state.yaml")

        if changes_made:
            print(
                f"Job '{self.name}' prepared for rerun! Removed: {', '.join(changes_made)}"
            )
        else:
            print(f"Job '{self.name}' prepared for rerun! (No cleanup needed)")

    # ──────────────────────────────────────────────
    # Permissions (review/)
    # ──────────────────────────────────────────────

    def _get_perm_context(self):
        from syft_perms import SyftPermContext

        datasite = self._client.config.syftbox_folder / self.datasite_owner_email
        return SyftPermContext(datasite=datasite)

    def _relative_review_path(self, subpath: str) -> Path:
        """Return path relative to the datasite for a review/ subpath."""
        rel = self.job_review_path.relative_to(
            self._client.config.syftbox_folder / self.datasite_owner_email
        )
        return rel / subpath

    def share_outputs(self, users: list[str]) -> None:
        """Grant read access to the outputs directory for given users."""
        ctx = self._get_perm_context()
        outputs_rel = self._relative_review_path("outputs")
        folder = ctx.open(outputs_rel)
        for user in users:
            folder.grant_read_access(user)

    def share_logs(self, users: list[str]) -> None:
        """Grant read access to log files (stdout, stderr, returncode) for given users."""
        ctx = self._get_perm_context()
        for filename in ("stdout.txt", "stderr.txt", "returncode.txt"):
            file_rel = self._relative_review_path(filename)
            f = ctx.open(file_rel)
            for user in users:
                f.grant_read_access(user)

    # ──────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        status_emojis = {
            "received": "📨",
            "pending": "📥",
            "approved": "✅",
            "rejected": "❌",
            "running": "🔄",
            "done": "🎉",
            "failed": "💥",
        }
        emoji = status_emojis.get(self.status, "❓")
        approval_info = ""
        if self.approval_method:
            approval_info = f" [approved: {self.approval_method}]"
        base = f"{emoji} {self.name} ({self.status}{approval_info}) -> {self.datasite_owner_email}"
        if self.review_reason:
            base += f" | Review reason: {self.review_reason}"
        return base

    def __repr__(self) -> str:
        parts = [
            f"name='{self.name}'",
            f"submitted_by='{self.submitted_by}'",
            f"current_user_email='{self.current_user_email}'",
            f"status='{self.status}'",
        ]
        if self.approval_method:
            parts.append(f"approval_method='{self.approval_method}'")
        if self.review_reason:
            parts.append(f"review_reason='{self.review_reason}'")
        return f"JobInfo({', '.join(parts)})"

    def _repr_html_(self) -> str:
        return job_info_repr_html(self)


class JobsList:
    """A list-like container for JobInfo objects with nice display."""

    def __init__(self, jobs: List[JobInfo], root_email: str, has_do_role: bool = False):
        self._jobs = jobs
        self._root_email = root_email
        self._has_do_role = has_do_role

    def __getitem__(self, index: int | str) -> JobInfo:
        if isinstance(index, int):
            return self._jobs[index]
        elif isinstance(index, str):
            for job in self._jobs:
                if job.name == index:
                    return job
            raise ValueError(f"Job with name '{index}' not found")
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self):
        return iter(self._jobs)

    def __str__(self) -> str:
        return jobs_list_str(self._jobs, self._root_email, self._has_do_role)

    def __repr__(self) -> str:
        return f"JobsList({len(self._jobs)} jobs)"

    def _repr_html_(self) -> str:
        return jobs_list_repr_html(self._jobs, self._root_email, self._has_do_role)
