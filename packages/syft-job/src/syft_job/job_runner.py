import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import psutil

from .client import JobClient
from .job import JobInfo
from . import __version__
from .config import SyftJobConfig
from .models.state import JobState, JobStatus
from .models.config import JobSubmissionMetadata

# Default timeout for job execution (10 minutes)
DEFAULT_JOB_TIMEOUT_SECONDS = 600


def get_job_timeout_seconds() -> int:
    """Get job timeout from environment variable or use default.

    Can be overridden by setting SYFT_DEFAULT_JOB_TIMEOUT_SECONDS environment variable.
    """
    return int(
        os.environ.get("SYFT_DEFAULT_JOB_TIMEOUT_SECONDS", DEFAULT_JOB_TIMEOUT_SECONDS)
    )


IS_IN_JOB_ENV_VAR = "SYFT_IS_IN_JOB"


def _kill_process_tree(pid: int, timeout: float = 2.0) -> None:
    """Kill `pid` and every descendant. Cross-platform via psutil."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=timeout)


class SyftJobRunner:
    """Job runner that monitors and executes approved jobs.

    Reads run.sh from inbox/, writes all output artifacts to review/.
    """

    def __init__(self, config: SyftJobConfig, poll_interval: int = 5):
        """
        Initialize the job runner.

        Args:
            config: SyftJobConfig instance
            poll_interval: How often to check for new jobs (in seconds)
        """
        self.config = config
        self.poll_interval = poll_interval
        self.known_jobs: Set[str] = set()

        # Ensure directory structure exists for the root user
        self._ensure_root_user_directories()

    @classmethod
    def from_config(cls, config: SyftJobConfig) -> "SyftJobRunner":
        return cls(config)

    def _ensure_root_user_directories(self) -> None:
        """Ensure inbox and review directory structure exists for the root user."""
        root_email = self.config.current_user_email
        inbox_dir = self.config.get_all_submissions_dir(root_email)
        review_dir = self.config.get_review_dir(root_email)

        inbox_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directories exist: {inbox_dir.parent}")

    def _get_pending_jobs(self) -> List[str]:
        """Get list of job paths (ds_email/job_name) in pending status."""
        review_dir = self.config.get_review_dir(self.config.current_user_email)

        if not review_dir.exists():
            return []

        jobs = []
        for ds_dir in review_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for job_dir in ds_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                state_file = job_dir / "state.yaml"
                if not state_file.exists():
                    continue
                try:
                    state = JobState.load(state_file)
                    if state.status == JobStatus.PENDING:
                        jobs.append(f"{ds_dir.name}/{job_dir.name}")
                except Exception:
                    continue
        return jobs

    def _print_new_job(self, job_name: str) -> None:
        """Print information about a new job in the inbox."""
        inbox_dir = self.config.get_all_submissions_dir(self.config.current_user_email)
        job_dir = inbox_dir / job_name

        print(f"\n NEW JOB DETECTED: {job_name}")
        print(f" Location: {job_dir}")

        # Check if run.sh exists and show first few lines
        run_script = job_dir / "run.sh"
        if run_script.exists():
            try:
                with open(run_script, "r") as f:
                    all_lines = f.readlines()
                lines = all_lines[:5]  # Show first 5 lines
                print(" Script preview:")
                for i, line in enumerate(lines, 1):
                    print(f"   {i}: {line.rstrip()}")
                if len(all_lines) > 5:
                    print("   ... (more lines)")
            except Exception as e:
                print(f"   Could not read script: {e}")

        # Check if config.yaml exists and show contents
        config_file = job_dir / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    content = f.read()
                print(" Config:")
                for line in content.split("\n"):
                    if line.strip():
                        print(f"   {line}")
            except Exception as e:
                print(f"   Could not read config: {e}")

        print("-" * 50)

    def reset_all_jobs(self) -> None:
        """
        Delete all jobs and recreate the job folder structure.

        This will:
        1. Delete all jobs in inbox and review
        2. Recreate the empty folder structure
        3. Reset the known jobs tracking
        """
        root_email = self.config.current_user_email
        job_dir = self.config.get_job_dir(root_email)

        print(f"RESETTING ALL JOBS for {root_email}")
        print(f" Target directory: {job_dir}")

        if not job_dir.exists():
            print(" No job directory found - nothing to reset")
            self._ensure_root_user_directories()
            return

        # Count jobs before deletion
        total_jobs = 0
        inbox_dir = self.config.get_all_submissions_dir(root_email)
        review_dir = self.config.get_review_dir(root_email)

        for scan_dir in [inbox_dir, review_dir]:
            if scan_dir.exists():
                for ds_dir in scan_dir.iterdir():
                    if ds_dir.is_dir():
                        for item in ds_dir.iterdir():
                            if item.is_dir():
                                total_jobs += 1

        if total_jobs == 0:
            print(" No jobs found to delete")
            self._ensure_root_user_directories()
            return

        # Confirm deletion
        print(f"\n WARNING: This will permanently delete {total_jobs} jobs!")
        print("   This action cannot be undone.")

        try:
            # Delete the entire job directory
            print(f" Deleting job directory: {job_dir}")
            shutil.rmtree(job_dir)

            # Recreate the folder structure
            print(" Recreating job folder structure...")
            self._ensure_root_user_directories()

            # Reset known jobs tracking
            self.known_jobs.clear()

            print(" Job reset completed successfully!")
            print(f"    - Deleted {total_jobs} jobs total")
            print("    - Clean job directory recreated")

        except Exception as e:
            print(f" Error during reset: {e}")
            print(" Attempting to recreate job directory anyway...")
            try:
                self._ensure_root_user_directories()
                print(" Job directory recreated")
            except Exception as recovery_error:
                print(f" Failed to recreate job directory: {recovery_error}")
                raise

    def check_for_new_jobs(self) -> None:
        """Check for new jobs in pending status and print them."""
        current_jobs = set(self._get_pending_jobs())
        new_jobs = current_jobs - self.known_jobs

        for job_name in new_jobs:
            self._print_new_job(job_name)

        # Update known jobs
        self.known_jobs = current_jobs

    def _get_jobs_in_approved(self) -> List[str]:
        """Get list of job paths (ds_email/job_name) in approved status."""
        review_dir = self.config.get_review_dir(self.config.current_user_email)

        if not review_dir.exists():
            return []

        jobs = []
        for ds_dir in review_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for job_dir in ds_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                state_file = job_dir / "state.yaml"
                if not state_file.exists():
                    continue
                try:
                    state = JobState.load(state_file)
                    if state.status == JobStatus.APPROVED:
                        jobs.append(f"{ds_dir.name}/{job_dir.name}")
                except Exception:
                    continue
        return jobs

    def _resolve_submission_dir(self, job_name: str, user: str | None = None) -> Path:
        """Resolve the inbox directory for a job."""
        if user:
            return self.config.get_job_submission_dir(
                self.config.current_user_email, user, job_name
            )
        inbox_dir = self.config.get_all_submissions_dir(self.config.current_user_email)
        matches = list(inbox_dir.glob(f"*/{job_name}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Job '{job_name}' not found in inbox under {inbox_dir}"
            )
        raise ValueError(
            f"Multiple jobs named '{job_name}' found: {matches}. "
            "Pass user= to disambiguate."
        )

    def _resolve_review_dir(self, job_name: str, user: str | None = None) -> Path:
        """Resolve the review directory for a job."""
        review_dir = self.config.get_review_dir(self.config.current_user_email)
        if user:
            return review_dir / user / job_name
        matches = list(review_dir.glob(f"*/{job_name}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Job '{job_name}' not found in review under {review_dir}"
            )
        raise ValueError(
            f"Multiple jobs named '{job_name}' found: {matches}. "
            "Pass user= to disambiguate."
        )

    def _execute_job_streaming(
        self, job_name: str, timeout: int, user: str | None = None
    ) -> int:
        """Execute job with real-time streaming output.

        Reads run.sh from inbox/, writes stdout/stderr to review/.
        """
        submission_dir = self._resolve_submission_dir(job_name, user)
        review_dir = self._resolve_review_dir(job_name, user)
        run_script = submission_dir / "run.sh"

        # Log prefix for streaming output
        log_prefix = f"[{self.config.current_user_email}][{job_name}]"

        # Make run.sh executable
        os.chmod(run_script, 0o755)

        # Prepare environment variables
        env = os.environ.copy()
        env["SYFTBOX_FOLDER"] = self.config.syftbox_folder_path_str
        env["SYFTBOX_EMAIL"] = self.config.current_user_email
        env[IS_IN_JOB_ENV_VAR] = "true"
        env["PYTHONUNBUFFERED"] = "1"

        # stdout/stderr go to review/
        stdout_file = review_dir / "stdout.txt"
        stderr_file = review_dir / "stderr.txt"

        import selectors

        with (
            open(stdout_file, "w") as stdout_f,
            open(stderr_file, "w") as stderr_f,
        ):
            process = subprocess.Popen(
                ["bash", str(run_script)],
                cwd=submission_dir,  # run.sh executes from inbox/ where code/ lives
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ, data="stdout")
            sel.register(process.stderr, selectors.EVENT_READ, data="stderr")

            start_time = time.time()
            timed_out = False

            # Stream output while process is running
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    _kill_process_tree(process.pid)
                    process.wait()
                    timed_out = True
                    print(f" Job {job_name} timed out after {timeout // 60} minutes")
                    stdout_f.write("\n--- PROCESS TIMED OUT ---\n")
                    stderr_f.write("\n--- PROCESS TIMED OUT ---\n")
                    break

                for key, _ in sel.select(timeout=0.1):
                    line = key.fileobj.readline()
                    if line:
                        if key.data == "stdout":
                            print(f"{log_prefix} {line}", end="", flush=True)
                            stdout_f.write(line)
                        else:
                            print(f"{log_prefix} STDERR: {line}", end="", flush=True)
                            stderr_f.write(line)

            sel.close()

            # Process exited - drain any remaining data from pipes
            remaining_stdout = process.stdout.read()
            remaining_stderr = process.stderr.read()

            if remaining_stdout:
                for line in remaining_stdout.splitlines(keepends=True):
                    print(f"{log_prefix} {line}", end="", flush=True)
                    stdout_f.write(line)

            if remaining_stderr:
                for line in remaining_stderr.splitlines(keepends=True):
                    print(f"{log_prefix} STDERR: {line}", end="", flush=True)
                    stderr_f.write(line)

            returncode = process.returncode if not timed_out else -1

        return returncode

    def _execute_job_captured(
        self, job_name: str, timeout: int, user: str | None = None
    ) -> int:
        """Execute job with captured output (non-streaming).

        Reads run.sh from inbox/, writes stdout/stderr to review/.
        """
        submission_dir = self._resolve_submission_dir(job_name, user)
        review_dir = self._resolve_review_dir(job_name, user)
        run_script = submission_dir / "run.sh"

        # Make run.sh executable
        os.chmod(run_script, 0o755)

        # Prepare environment variables
        env = os.environ.copy()
        env["SYFTBOX_FOLDER"] = self.config.syftbox_folder_path_str
        env["SYFTBOX_EMAIL"] = self.config.current_user_email
        env[IS_IN_JOB_ENV_VAR] = "true"
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            ["bash", str(run_script)],
            cwd=submission_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            _kill_process_tree(process.pid)
            stdout, stderr = process.communicate()
            returncode = -1
            stdout = (stdout or "") + "\n--- PROCESS TIMED OUT ---\n"
            stderr = (stderr or "") + "\n--- PROCESS TIMED OUT ---\n"
            print(f" Job {job_name} timed out after {timeout // 60} minutes")

        stdout_file = review_dir / "stdout.txt"
        with open(stdout_file, "w") as f:
            f.write(stdout)

        stderr_file = review_dir / "stderr.txt"
        with open(stderr_file, "w") as f:
            f.write(stderr)

        return returncode

    def _execute_job(
        self,
        job_name: str,
        stream_output: bool = True,
        timeout: int | None = None,
        user: str | None = None,
    ) -> bool:
        """
        Execute run.sh for an approved job.

        Reads run.sh from inbox/, writes all output to review/.

        Args:
            job_name: Name of the job to execute.
            stream_output: If True (default), stream output in real-time.
            timeout: Timeout in seconds. Defaults to 300 (5 minutes).
            user: DS email who submitted the job. If None, searches.

        Returns:
            bool: True if execution was successful, False otherwise
        """
        if timeout is None:
            timeout = get_job_timeout_seconds()

        submission_dir = self._resolve_submission_dir(job_name, user)
        review_dir = self._resolve_review_dir(job_name, user)
        run_script = submission_dir / "run.sh"

        if not run_script.exists():
            print(f" No run.sh found in {job_name}")
            return False

        self._prepare_outputs_dir(job_name, user)

        print(f" Executing job: {job_name}")
        print(f" Inbox: {submission_dir}")

        # Update state to RUNNING
        state_file = review_dir / "state.yaml"
        state = JobState.load(state_file)
        state.status = JobStatus.RUNNING
        state.save(state_file)

        try:
            if stream_output:
                returncode = self._execute_job_streaming(job_name, timeout, user)
            else:
                returncode = self._execute_job_captured(job_name, timeout, user)

            # Move outputs from inbox/ to review/
            self._move_outputs_to_review(submission_dir, review_dir)

            # Write return code to review/
            returncode_file = review_dir / "returncode.txt"
            with open(returncode_file, "w") as f:
                f.write(str(returncode))

            # Update state to DONE or FAILED
            self._set_finalized_job_state(state_file, returncode)

            stdout_file = review_dir / "stdout.txt"
            stderr_file = review_dir / "stderr.txt"

            if returncode == 0:
                print(f" Job {job_name} completed successfully")
                print(f" Output written to {stdout_file}")
            else:
                print(f" Job {job_name} completed with return code {returncode}")
                print(f" Output written to {stdout_file}")
                try:
                    if stderr_file.exists() and stderr_file.stat().st_size > 0:
                        print(f" Error output written to {stderr_file}")
                except OSError:
                    pass

            return True

        except subprocess.TimeoutExpired:
            print(f" Job {job_name} timed out after {timeout // 60} minutes")
            self._set_finalized_job_state(state_file, -1)
            return False
        except Exception as e:
            print(f" Error executing job {job_name}: {e}")
            self._set_finalized_job_state(state_file, -1)
            return False

    def _set_finalized_job_state(self, state_file: Path, returncode: int) -> None:
        state = JobState.load(state_file)
        state.status = JobStatus.DONE if returncode == 0 else JobStatus.FAILED
        state.completed_at = datetime.now(timezone.utc)
        state.return_code = returncode
        state.save(state_file)

    def _move_outputs_to_review(self, submission_dir: Path, review_dir: Path) -> None:
        inbox_outputs = submission_dir / "code" / "outputs"
        review_outputs = review_dir / "outputs"
        if inbox_outputs.exists() and inbox_outputs.is_dir():
            # Merge into review/outputs (which was pre-created by _prepare_outputs_dir)
            for item in inbox_outputs.iterdir():
                dest = review_outputs / item.name
                if item.is_file():
                    shutil.copy2(str(item), str(dest))
                elif item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(str(item), str(dest))
            # Clean up inbox outputs
            shutil.rmtree(inbox_outputs)

    def _prepare_outputs_dir(self, job_name: str, user: str | None = None) -> None:
        """Clear and recreate outputs dir in both inbox/ (for job cwd) and review/ (for final results)."""
        # Create outputs/ inside code/ dir so job scripts can write there (cwd is code/)
        submission_dir = self._resolve_submission_dir(job_name, user)
        inbox_outputs = submission_dir / "code" / "outputs"
        inbox_outputs.mkdir(parents=True, exist_ok=True)

        # Create outputs/ in review dir with owner-only read permissions
        review_dir = self._resolve_review_dir(job_name, user)
        outputs_dir = review_dir / "outputs"
        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        from syft_perms import SyftPermContext

        datasite = self.config.syftbox_folder / self.config.current_user_email
        rel_path = str(outputs_dir.relative_to(datasite)) + "/"
        ctx = SyftPermContext(datasite=datasite)
        folder = ctx.open(rel_path)
        folder.grant_read_access(self.config.current_user_email)

    def _get_job_submitter(self, job_name: str, user: str | None = None) -> str | None:
        """Read submitted_by from job config.yaml in inbox/."""
        metadata = self._get_job_metadata(job_name, user)
        if metadata is None:
            return None
        return metadata.submitted_by

    def _get_job_metadata(
        self, job_name: str, user: str | None = None
    ) -> JobSubmissionMetadata:
        submission_dir = self._resolve_submission_dir(job_name, user)
        config_file = submission_dir / "config.yaml"
        if not config_file.exists():
            return None
        try:
            return JobSubmissionMetadata.load(config_file)
        except Exception:
            return None

    def _get_job_state(self, job_name: str, user: str | None = None) -> JobState:
        review_dir = self._resolve_review_dir(job_name, user)
        state_file = review_dir / "state.yaml"
        if state_file.exists():
            return JobState.load(state_file)
        return JobState(status=JobStatus.RECEIVED)

    def _get_job_info(self, job_name: str, user: str | None = None):
        """Create a JobInfo for a job by name."""
        metadata = self._get_job_metadata(job_name, user)
        if metadata is None:
            raise ValueError(f"Job '{job_name}' not found")

        state = self._get_job_state(job_name, user)
        client = JobClient(config=self.config)
        return JobInfo(
            job_metadata=metadata,
            state=state,
            datasite_owner_email=self.config.current_user_email,
            current_user_email=self.config.current_user_email,
            client=client,
        )

    def process_approved_jobs(
        self,
        stream_output: bool = True,
        timeout: int | None = None,
        skip_job_names: list[str] | None = None,
        share_outputs_with_submitter: bool = False,
        share_logs_with_submitter: bool = False,
    ) -> None:
        """Process all jobs in approved status.

        Args:
            stream_output: If True (default), stream output in real-time.
            timeout: Timeout in seconds per job. Defaults to 300 (5 minutes).
            skip_job_names: Optional list of job names to skip.
            share_outputs_with_submitter: If True, grant read access on outputs to submitter.
            share_logs_with_submitter: If True, grant read access on logs to submitter.
        """
        approved_jobs = self._get_jobs_in_approved()

        if not approved_jobs:
            return

        # Filter out jobs to skip
        if skip_job_names:
            skip_set = set(skip_job_names)
            approved_jobs = [j for j in approved_jobs if j not in skip_set]

        if not approved_jobs:
            return

        print(f" Found {len(approved_jobs)} job(s) in approved status")

        for job_path in approved_jobs:
            # job_path is "{ds_email}/{job_name}"
            user, job_name = job_path.split("/", 1)
            print(f"\n{'=' * 50}")
            self._execute_job(
                job_name, stream_output=stream_output, timeout=timeout, user=user
            )
            self.share_job_results(
                job_name,
                share_outputs_with_submitter,
                share_logs_with_submitter,
                user=user,
            )
            print(f"{'=' * 50}")

        if approved_jobs:
            print(f"\n Processed {len(approved_jobs)} job(s)")

    def share_job_results(
        self,
        job_name: str,
        share_outputs: bool,
        share_logs: bool,
        user: str | None = None,
    ) -> None:
        """Share job outputs/logs with submitter if requested."""
        if not share_outputs and not share_logs:
            return
        submitter = self._get_job_submitter(job_name, user)
        if not submitter:
            return
        job_info = self._get_job_info(job_name, user)
        if share_outputs:
            job_info.share_outputs([submitter])
        if share_logs:
            job_info.share_logs([submitter])

    def run(self) -> None:
        """Start monitoring the inbox and approved folders for jobs."""
        root_email = self.config.current_user_email
        job_dir = self.config.get_job_dir(root_email)

        print(f" SyftJob Runner started: version: {__version__}")
        print(f" Monitoring jobs for: {root_email}")
        print(f" Job directory: {job_dir}")
        print(f" Poll interval: {self.poll_interval} seconds")
        print(" Press Ctrl+C to stop")
        print("=" * 50)

        # Initialize known jobs with current state
        self.known_jobs = set(self._get_pending_jobs())
        if self.known_jobs:
            print(
                f" Found {len(self.known_jobs)} existing pending jobs: "
                f"{', '.join(self.known_jobs)}"
            )
        else:
            print(" No existing pending jobs found")
        print("-" * 50)

        try:
            while True:
                # Scan inbox for new submissions and receive them
                from .client import JobClient

                client = JobClient(config=self.config)
                client.scan_inbox()

                self.check_for_new_jobs()
                self.process_approved_jobs()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n Job runner stopped by user")
        except Exception as e:
            print(f"\n Job runner encountered an error: {e}")
            raise


def create_runner(
    syftbox_folder_path: str, email: str, poll_interval: int = 5
) -> SyftJobRunner:
    """
    Factory function to create a SyftJobRunner from SyftBox folder.

    Args:
        syftbox_folder_path: Path to the SyftBox folder
        email: Email address of the user
        poll_interval: How often to check for new jobs (in seconds)

    Returns:
        Configured SyftJobRunner instance
    """
    config = SyftJobConfig.from_syftbox_folder(syftbox_folder_path, email)
    return SyftJobRunner(config, poll_interval)
