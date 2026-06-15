"""Job event handler for notifications."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from syft_bg.common.state import JsonStateManager
from syft_bg.notify.gmail.sender import GmailSender
from syft_client.sync.utils.path_filters import is_normal_syncable_path

if TYPE_CHECKING:
    from syft_job.client import JobClient

_MAX_FILE_SIZE = 50 * 1024  # 50 KB per file
_MAX_TOTAL_CODE_SIZE = 500 * 1024  # 500 KB total code in email
_MAX_STDERR_SIZE = 50 * 1024  # 50 KB stderr


def _read_job_code(job_client: "JobClient", job_name: str) -> Optional[dict[str, str]]:
    """Read job code contents as a dict of filename -> file contents."""
    job = next((j for j in job_client.jobs if j.name == job_name), None)
    if not job or not job.code_dir.exists():
        return None

    code_files: dict[str, str] = {}
    total_size = 0
    all_files = sorted(f for f in job.code_dir.rglob("*") if f.is_file())

    datasite_root = (
        job_client.config.syftbox_folder / job_client.config.current_user_email
    )

    for f in all_files:
        rel = str(f.relative_to(job.code_dir))
        datasite_rel = f.relative_to(datasite_root)
        if not is_normal_syncable_path(datasite_rel):
            continue
        try:
            content = f.read_text(errors="replace")
        except Exception as e:
            code_files[rel] = f"[unable to read file: {e}]"
            continue

        file_size = len(content.encode("utf-8"))

        if file_size > _MAX_FILE_SIZE:
            truncated = content[:_MAX_FILE_SIZE]
            content = (
                f"{truncated}\n\n[... truncated, full file is {file_size // 1024} KB]"
            )
            file_size = _MAX_FILE_SIZE

        if total_size + file_size > _MAX_TOTAL_CODE_SIZE:
            remaining = len(all_files) - len(code_files)
            code_files["..."] = f"[... {remaining} more files not shown]"
            break

        code_files[rel] = content
        total_size += file_size

    return code_files if code_files else None


def _read_job_stderr(
    job_client: "JobClient", job_name: str
) -> tuple[Optional[str], Optional[int]]:
    """Read stderr and return code from a job's review directory."""
    job = next((j for j in job_client.jobs if j.name == job_name), None)
    if not job:
        return None, None

    stderr_text = None
    stderr_file = job.job_review_path / "stderr.txt"
    if stderr_file.exists():
        try:
            file_size = stderr_file.stat().st_size
            if file_size > _MAX_STDERR_SIZE:
                with open(stderr_file, "rb") as fh:
                    fh.seek(-_MAX_STDERR_SIZE, 2)
                    tail = fh.read().decode("utf-8", errors="replace")
                stderr_text = (
                    f"[... truncated, showing last {_MAX_STDERR_SIZE // 1024} KB "
                    f"of {file_size // 1024} KB total]\n\n{tail}"
                )
            else:
                stderr_text = stderr_file.read_text(errors="replace").strip() or None
        except Exception:
            pass

    return_code = None
    rc_file = job.job_review_path / "returncode.txt"
    if rc_file.exists():
        try:
            return_code = int(rc_file.read_text().strip())
        except (ValueError, OSError):
            pass

    return stderr_text, return_code


def _friendly_reason(reason: str, job_name: str) -> str:
    """Convert internal rejection reason to a DS-friendly message."""
    if "unknown peer" in reason:
        return (
            f'Your job "{job_name}" could not be processed because your account '
            "is not yet registered with this data owner. Please contact them to "
            "get added as an approved collaborator."
        )
    if "unapproved file" in reason:
        return (
            f'Your job "{job_name}" contained a script file that hasn\'t been '
            "approved by the data owner. Please check that you're submitting "
            "the correct files."
        )
    if "hash mismatch" in reason:
        return (
            f'Your job "{job_name}" contained a script that differs from the '
            "approved version. This can happen if the file was modified after "
            "approval. Please contact the data owner to re-approve your script."
        )
    if "no Python files" in reason:
        return (
            f'Your job "{job_name}" did not contain any Python files. '
            "Please make sure your submission includes the required scripts."
        )
    return f'Your job "{job_name}" was not approved: {reason}'


class JobHandler:
    """Handles job-related notification events."""

    def __init__(
        self,
        sender: GmailSender,
        state: JsonStateManager,
        do_email: str = "",
        syftbox_root: Optional[Path] = None,
        notify_on_new: bool = True,
        notify_on_approved: bool = True,
        notify_on_executed: bool = True,
    ):
        self.sender = sender
        self.state = state
        self.do_email = do_email
        self.notify_on_new = notify_on_new
        self.notify_on_approved = notify_on_approved
        self.notify_on_executed = notify_on_executed

        self.job_client = None
        if syftbox_root and do_email:
            from syft_job import SyftJobConfig
            from syft_job.client import JobClient

            config = SyftJobConfig(
                syftbox_folder=syftbox_root, current_user_email=do_email
            )
            self.job_client = JobClient.from_config(config)

    def on_new_job(
        self,
        do_email: str,
        job_name: str,
        submitter: str,
        job_url: Optional[str] = None,
    ) -> bool:
        """Notify DO about a new job."""
        if not self.notify_on_new:
            return False

        if self.state.was_notified(job_name, "new"):
            print(f"[JobHandler] Skip {job_name}/new: already notified")
            return False

        job_code = None
        if self.job_client:
            job_code = _read_job_code(self.job_client, job_name)

        result = self.sender.notify_new_job(
            do_email,
            job_name,
            submitter,
            job_url=job_url,
            job_code=job_code,
        )

        if result.success:
            self.state.mark_notified(job_name, "new")
            if result.thread_id:
                self.state.store_thread_id(job_name, result.thread_id)
            self.sender.notify_job_submitted_to_ds(submitter, job_name, do_email)
        else:
            import traceback

            print(
                f"[JobHandler] Failed to send new job notification for {job_name}: {traceback.format_exc()}"
            )

        return result.success

    def on_job_approved(
        self,
        ds_email: str,
        job_name: str,
        job_url: Optional[str] = None,
    ) -> bool:
        """Notify DS that their job was approved, and DO in the same thread."""
        if not self.notify_on_approved:
            return False

        if self.state.was_notified(job_name, "approved"):
            print(f"[JobHandler] Skip {job_name}/approved: already notified")
            return False

        result = self.sender.notify_job_approved(ds_email, job_name, job_url=job_url)

        if result.success:
            self.state.mark_notified(job_name, "approved")
            # Also notify DO in the same thread
            if self.do_email:
                thread_id = self.state.get_thread_id(job_name)
                self.sender.notify_job_approved_to_do(
                    self.do_email, job_name, ds_email, thread_id=thread_id
                )
        else:
            print(
                f"[JobHandler] Failed to send approved notification for {job_name} to {ds_email}"
            )

        return result.success

    def on_job_executed(
        self,
        ds_email: str,
        job_name: str,
        duration: Optional[int] = None,
        results_url: Optional[str] = None,
    ) -> bool:
        """Notify DS that their job finished, and DO in the same thread."""
        if not self.notify_on_executed:
            return False

        if self.state.was_notified(job_name, "executed"):
            print(f"[JobHandler] Skip {job_name}/executed: already notified")
            return False

        result = self.sender.notify_job_executed(
            ds_email, job_name, duration=duration, results_url=results_url
        )

        if result.success:
            self.state.mark_notified(job_name, "executed")
            # Also notify DO in the same thread
            if self.do_email:
                thread_id = self.state.get_thread_id(job_name)
                self.sender.notify_job_completed_to_do(
                    self.do_email,
                    job_name,
                    ds_email,
                    duration=duration,
                    thread_id=thread_id,
                )
        else:
            print(
                f"[JobHandler] Failed to send executed notification for {job_name} to {ds_email}"
            )

        return result.success

    def on_job_failed(
        self,
        ds_email: str,
        job_name: str,
        duration: Optional[int] = None,
    ) -> bool:
        """Notify DO that a job failed during execution."""
        if not self.notify_on_executed:
            return False

        if self.state.was_notified(job_name, "failed"):
            print(f"[JobHandler] Skip {job_name}/failed: already notified")
            return False

        error_output, return_code = None, None
        if self.job_client:
            error_output, return_code = _read_job_stderr(self.job_client, job_name)

        if self.do_email:
            thread_id = self.state.get_thread_id(job_name)
            result = self.sender.notify_job_failed_to_do(
                self.do_email,
                job_name,
                ds_email,
                error_output=error_output,
                return_code=return_code,
                duration=duration,
                thread_id=thread_id,
            )

            if result.success:
                self.state.mark_notified(job_name, "failed")
            else:
                print(f"[JobHandler] Failed to send failed notification for {job_name}")

            return result.success

        return False

    def on_job_rejected(
        self,
        do_email: str,
        job_name: str,
        ds_email: str,
        reason: str,
    ) -> bool:
        """Notify DO and DS that a job was rejected."""
        if self.state.was_notified(job_name, "rejected"):
            print(f"[JobHandler] Skip {job_name}/rejected: already notified")
            return False

        thread_id = self.state.get_thread_id(job_name)
        result = self.sender.notify_job_rejected_to_do(
            do_email, job_name, ds_email, reason, thread_id=thread_id
        )

        if result.success:
            self.state.mark_notified(job_name, "rejected")
            # Also notify DS with a friendly message
            friendly = _friendly_reason(reason, job_name)
            self.sender.notify_job_rejected_to_ds(ds_email, job_name, friendly)
        else:
            print(f"[JobHandler] Failed to send rejected notification for {job_name}")

        return result.success
