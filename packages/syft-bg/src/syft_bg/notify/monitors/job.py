"""Job monitor for detecting new jobs and status changes."""

from pathlib import Path
from typing import Optional

from syft_job.config import SyftJobConfig
from syft_job.models.config import JobSubmissionMetadata
from syft_job.models.state import JobState, JobStatus

from syft_bg.common.monitor import Monitor
from syft_bg.common.state import JsonStateManager
from syft_bg.notify.handlers.job import JobHandler


class JobMonitor(Monitor):
    """Monitors for new jobs and job status changes via local filesystem."""

    def __init__(
        self,
        syftbox_root: Path,
        do_email: str,
        handler: JobHandler,
        state: JsonStateManager,
    ):
        super().__init__()
        self.syftbox_root = Path(syftbox_root).expanduser()
        self.do_email = do_email
        self.handler = handler
        self.state = state
        self.job_config = SyftJobConfig.from_syftbox_folder(
            str(self.syftbox_root), do_email
        )

    def _check_all_entities(self):
        self.process_local_status_changes()

    def process_local_status_changes(self):
        inbox_dir = self.job_config.get_all_submissions_dir(self.do_email)
        if not inbox_dir.exists():
            return

        for ds_dir in inbox_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for job_path in ds_dir.iterdir():
                if not job_path.is_dir():
                    continue
                try:
                    self._maybe_process_job(job_path)
                except Exception as e:
                    print(f"[JobMonitor] Error checking job {job_path.name}: {e}")

    def _maybe_process_job(self, job_path: Path):
        metadata = self._load_job_metadata(job_path)
        if not metadata:
            return

        job_name = metadata.name
        ds_email = metadata.submitted_by

        if not self.state.was_notified(job_name, "new"):
            success = self.handler.on_new_job(self.do_email, job_name, ds_email)
            if success:
                print(f"[JobMonitor] Sent new job notification: {job_name}")

        review_state = self._load_review_state(ds_email, job_name)

        if review_state and review_state.status in (
            JobStatus.APPROVED,
            JobStatus.RUNNING,
            JobStatus.DONE,
            JobStatus.FAILED,
        ):
            success = self.handler.on_job_approved(ds_email, job_name)
            if success:
                print(f"[JobMonitor] Sent job approved notification: {job_name}")

        if review_state and review_state.status == JobStatus.FAILED:
            success = self.handler.on_job_failed(ds_email, job_name)
            if success:
                print(f"[JobMonitor] Sent job failed notification: {job_name}")
        elif review_state and review_state.status == JobStatus.DONE:
            success = self.handler.on_job_executed(ds_email, job_name)
            if success:
                print(f"[JobMonitor] Sent job executed notification: {job_name}")

    def seed_existing_jobs(self):
        """On fresh state, mark all existing jobs so we don't re-notify old jobs."""
        inbox_dir = self.job_config.get_all_submissions_dir(self.do_email)
        if not inbox_dir.exists():
            return

        count = 0
        for ds_dir in inbox_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for job_path in ds_dir.iterdir():
                if not job_path.is_dir():
                    continue
                metadata = self._load_job_metadata(job_path)
                if not metadata:
                    continue
                self.state.mark_notified(metadata.name, "new")
                review_state = self._load_review_state(
                    metadata.submitted_by, metadata.name
                )
                if review_state:
                    if review_state.status in (
                        JobStatus.APPROVED,
                        JobStatus.RUNNING,
                        JobStatus.DONE,
                        JobStatus.FAILED,
                    ):
                        self.state.mark_notified(metadata.name, "approved")
                    if review_state.status == JobStatus.DONE:
                        self.state.mark_notified(metadata.name, "executed")
                    if review_state.status == JobStatus.FAILED:
                        self.state.mark_notified(metadata.name, "failed")
                count += 1

        if count:
            print(f"[JobMonitor] Seeded {count} existing jobs on fresh state")

    def _load_review_state(self, ds_email: str, job_name: str) -> Optional[JobState]:
        """Load state.yaml from the job's review directory."""
        review_dir = self.job_config.get_review_job_dir(
            self.do_email, ds_email, job_name
        )
        state_file = review_dir / "state.yaml"
        if not state_file.exists():
            return None
        try:
            return JobState.load(state_file)
        except Exception:
            return None

    def _load_job_metadata(self, job_path: Path) -> Optional[JobSubmissionMetadata]:
        config_file = job_path / "config.yaml"
        if not config_file.exists():
            return None

        try:
            return JobSubmissionMetadata.load(config_file)
        except Exception as e:
            print(f"[JobMonitor] Error reading job config {config_file}: {e}")
            return None
