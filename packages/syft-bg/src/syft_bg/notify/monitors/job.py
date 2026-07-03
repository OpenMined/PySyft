"""Job monitor for detecting new jobs and status changes."""

from pathlib import Path
from typing import Optional

from syft_job.config import SyftJobConfig
from syft_job.manager import JobManager, JobRef
from syft_job.models import JobState, JobStatus, JobSubmissionMetadata

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
        self.job_manager = JobManager(config=self.job_config)

    def _check_all_entities(self):
        self.process_local_status_changes()

    def process_local_status_changes(self):
        for ref in self.job_manager.iter_submission_refs(self.do_email):
            try:
                self._maybe_process_job(ref)
            except Exception as e:
                print(f"[JobMonitor] Error checking job {ref.job_name}: {e}")

    def _maybe_process_job(self, ref: JobRef):
        metadata = self._load_job_metadata(ref)
        if not metadata:
            return

        job_name = metadata.name
        ds_email = metadata.submitted_by

        if not self.state.was_notified(job_name, "new"):
            success = self.handler.on_new_job(self.do_email, job_name, ds_email)
            if success:
                print(f"[JobMonitor] Sent new job notification: {job_name}")

        review_state = self._load_review_state(ref)

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
        count = 0
        for ref in self.job_manager.iter_submission_refs(self.do_email):
            metadata = self._load_job_metadata(ref)
            if not metadata:
                continue
            self.state.mark_notified(metadata.name, "new")
            review_state = self._load_review_state(ref)
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

    def _load_review_state(self, ref: JobRef) -> Optional[JobState]:
        """Load state.yaml from the job's review directory."""
        try:
            return self.job_manager.read_state(ref)
        except Exception:
            return None

    def _load_job_metadata(self, ref: JobRef) -> Optional[JobSubmissionMetadata]:
        try:
            return self.job_manager.read_submission(ref)
        except Exception as e:
            print(f"[JobMonitor] Error reading job config for {ref.job_name}: {e}")
            return None
