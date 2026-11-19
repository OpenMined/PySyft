import logging
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import psutil
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from . import __version__
from .client import JobClient
from .config import SyftJobConfig
from .job import JobInfo
from .job_storage import JobRef, JobStateNotFoundError, JobStorage
from .models import JobState, JobStatus, JobSubmissionMetadata
from .policies import (
    OUTPUTS_WITHHELD_FILENAME,
    POLICY_ERROR_FILENAME,
    ApiResolver,
    JobPolicies,
    PolicyEnv,
    PolicyKind,
    PolicyResolutionError,
    PolicyResult,
    PolicyRunner,
    PolicyState,
    policy_stderr_filename,
)
from .read_permissions import (
    LOG_FILES,
    OUTPUTS,
    ReadPermissions,
    merge_read_permissions,
    validate_read_permissions,
)
from .review_grants import replace_ds_folder_read
from .submission import (
    InvalidSubmissionError,
    copy_submission,
    submission_hash,
)

# Default timeout for job execution (10 minutes)
DEFAULT_JOB_TIMEOUT_SECONDS = 600


def get_job_timeout_seconds() -> int:
    """Get job timeout from environment variable or use default.

    Can be overridden by setting SYFT_DEFAULT_JOB_TIMEOUT_SECONDS environment variable.
    """
    value = os.environ.get("SYFT_DEFAULT_JOB_TIMEOUT_SECONDS")
    if value is None:
        return DEFAULT_JOB_TIMEOUT_SECONDS

    try:
        timeout = int(value)
    except ValueError:
        timeout = 0

    if timeout <= 0:
        raise ValueError(
            "SYFT_DEFAULT_JOB_TIMEOUT_SECONDS must be a positive integer; "
            f"got {value!r}"
        )
    return timeout


IS_IN_JOB_ENV_VAR = "SYFT_IS_IN_JOB"


def _format_timeout(seconds: int) -> str:
    """Return a human-readable duration string for a timeout value.

    Sub-minute timeouts are expressed in seconds so they are never reported
    as "0 minutes" (issue #9527).

    Examples:
        >>> _format_timeout(30)
        '30 seconds'
        >>> _format_timeout(90)
        '1 minute 30 seconds'
        >>> _format_timeout(600)
        '10 minutes'
    """
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " ".join(parts)


# A job runs from a fresh copy of its approved submission in the system temp
# folder, outside the SyftBox: sync never reads it, and no submitter writes it.
RUN_DIR_PREFIX = "syft-job-run-"
RUN_DIR_NAME = "job"

NO_APPROVED_HASH_REASON = (
    "approved without a record of the approved submission; review and approve it again"
)
CHANGED_SUBMISSION_REASON = "the submission changed after it was approved"

logger = logging.getLogger(__name__)


def _read_params(run_dir: Path, params_file: str | None) -> bytes:
    """The call's params JSON from the run copy, {} when there is none."""
    path = run_dir / params_file if params_file else None
    return path.read_bytes() if path is not None and path.is_file() else b"{}"


def _read_call_files(run_dir: Path, call_files: tuple[str, ...]) -> dict[str, bytes]:
    """The files the call sent, from the run copy."""
    return {
        rel: (run_dir / rel).read_bytes()
        for rel in call_files
        if (run_dir / rel).is_file()
    }


def _remove_run_dir(run_dir: Path | None) -> None:
    """Remove a run copy and its temp folder; None is a no-op."""
    if run_dir is not None:
        shutil.rmtree(run_dir.parent, ignore_errors=True)


def _submitter_read_permissions(
    ds_email: str, share_logs: bool, share_outputs: bool
) -> ReadPermissions:
    """Read for the submitter on the logs and exit code, and on the outputs."""
    permissions = {name: [ds_email] for name in LOG_FILES} if share_logs else {}
    if share_outputs:
        permissions[OUTPUTS] = [ds_email]
    return permissions


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

    Reads run.sh from inbox/. State, outputs, logs and the return code go to
    review/. Only the DO reads a run file there until a share grants read.
    """

    def __init__(
        self,
        config: SyftJobConfig,
        poll_interval: int = 5,
        api_resolver: ApiResolver | None = None,
    ):
        """
        Initialize the job runner.

        Args:
            config: SyftJobConfig instance
            poll_interval: How often to check for new jobs (in seconds)
            api_resolver: Loads the api a request calls: its code and its
                policies. A request fails when there is no resolver.
        """
        self.config = config
        self.poll_interval = poll_interval
        self.api_resolver = api_resolver
        self.policy_runner = PolicyRunner()
        self.manager = JobStorage(config=config)
        self.known_jobs: Set[JobRef] = set()

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

    def _get_jobs_with_status(self, status: JobStatus) -> List[JobRef]:
        """Get refs of all jobs in review/ (any protocol layout) with ``status``."""
        jobs = []
        for ref in self.manager.iter_review_refs(self.config.current_user_email):
            try:
                state = self.manager.read_state(ref)
            except JobStateNotFoundError:
                continue
            if state is not None and state.status == status:
                jobs.append(ref)
        return jobs

    def _get_pending_jobs(self) -> List[JobRef]:
        """Get refs of jobs in pending status."""
        return self._get_jobs_with_status(JobStatus.PENDING)

    def _print_new_job(self, ref: JobRef) -> None:
        """Print information about a new job in the inbox."""
        job_dir = self.manager.submission_dir(ref)

        print(f"\n NEW JOB DETECTED: {ref.ds_email}/{ref.job_name}")
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
        total_jobs = len(list(self.manager.iter_submission_refs(root_email))) + len(
            list(self.manager.iter_review_refs(root_email))
        )

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

        for ref in new_jobs:
            self._print_new_job(ref)

        # Update known jobs
        self.known_jobs = current_jobs

    def _get_jobs_in_approved(self) -> List[JobRef]:
        """Get refs of jobs in approved status."""
        return self._get_jobs_with_status(JobStatus.APPROVED)

    def _find_jobref_from_name(self, job_name: str, user: str | None = None) -> JobRef:
        """Resolve the unique ref for a job by name (any protocol layout)."""
        return self.manager.find_submission_ref(
            self.config.current_user_email, job_name, ds_email=user
        )

    def _execute_job_streaming(self, ref: JobRef, timeout: int, run_dir: Path) -> int:
        """Execute job with real-time streaming output.

        Runs run.sh from ``run_dir``, writes stdout/stderr to review/.
        """
        run_script = run_dir / "run.sh"
        job_name = ref.job_name

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

        review_dir = self.manager.review_dir(ref)
        review_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = review_dir / "stdout.txt"
        stderr_file = review_dir / "stderr.txt"

        import selectors

        with (
            open(stdout_file, "w") as stdout_f,
            open(stderr_file, "w") as stderr_f,
        ):
            process = subprocess.Popen(
                ["bash", str(run_script)],
                cwd=run_dir,  # run.sh executes from the run copy where code/ lives
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            # Selector readiness only guarantees that some bytes are available.
            # Keep reads non-blocking so a partial line cannot stall timeout checks.
            os.set_blocking(process.stdout.fileno(), False)
            os.set_blocking(process.stderr.fileno(), False)

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
                    print(f" Job {job_name} timed out after {_format_timeout(timeout)}")
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

    def _execute_job_captured(self, ref: JobRef, timeout: int, run_dir: Path) -> int:
        """Execute job with captured output (non-streaming).

        Runs run.sh from ``run_dir``, writes stdout/stderr to review/.
        """
        run_script = run_dir / "run.sh"
        job_name = ref.job_name

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
            cwd=run_dir,
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
            print(f" Job {job_name} timed out after {_format_timeout(timeout)}")

        review_dir = self.manager.review_dir(ref)
        review_dir.mkdir(parents=True, exist_ok=True)

        stdout_file = review_dir / "stdout.txt"
        with open(stdout_file, "w") as f:
            f.write(stdout)

        stderr_file = review_dir / "stderr.txt"
        with open(stderr_file, "w") as f:
            f.write(stderr)

        return returncode

    def _execute_job_and_policies(
        self,
        ref: JobRef,
        stream_output: bool = True,
        timeout: int | None = None,
        approved_peers: list[str] | None = None,
        run_permissions: ReadPermissions | None = None,
    ) -> bool:
        """
        Execute run.sh of the submission the owner approved.

        The inbox is copied into a fresh folder outside the SyftBox, and the
        copy must match the approved hash before anything runs, so an edit of
        the submitter's inbox never runs. State, outputs, logs and the return
        code go to review/, where only the DO reads them until a share.

        Args:
            ref: Ref of the job to execute.
            stream_output: If True (default), stream output in real-time.
            timeout: Timeout in seconds. Defaults to the configured value or 600.
            approved_peers: The peers an output policy may give read permissions.
            run_permissions: Read permissions the run settings add to the
                approved ones, once the copy is verified and before the run.

        Returns:
            bool: True if run.sh ran, whatever its exit code. False if the job
            was not run: it has no approved hash, or the copy does not match it.
        """
        if timeout is None:
            timeout = get_job_timeout_seconds()

        run_dir = self._approved_run_dir(ref)
        if run_dir is None:
            return False
        try:
            resolved, policies = self._load_policies(ref)
            if resolved:
                self._add_approved(ref, run_permissions or {})
                self._run_approved_job(
                    ref, run_dir, stream_output, timeout, policies, approved_peers
                )
        finally:
            _remove_run_dir(run_dir)
        return resolved

    def _approved_run_dir(self, ref: JobRef) -> Path | None:
        """A fresh copy of the approved submission to run from, or None.

        No approved hash: the job was approved before hashes were recorded, or
        never approved (a deposited result that was rerun). It goes back to
        pending, to be approved again. A copy that does not match the approved
        hash fails the job. A local error while copying (disk, permissions,
        temp folder) leaves the job approved, so the next cycle retries it.
        """
        record = self.manager.read_owner_record(ref)
        if record is None or record.approved_hash is None:
            self._return_to_pending(ref)
            return None
        run_dir = None
        try:
            run_dir = self._copy_to_run_dir(ref)
            verified = submission_hash(run_dir) == record.approved_hash
        except InvalidSubmissionError:
            verified = False
        except OSError as e:
            _remove_run_dir(run_dir)
            logger.warning(f"Job {ref.job_name} not run, it stays approved: {e}")
            return None
        if not verified:
            _remove_run_dir(run_dir)
            self._fail_unverified(ref)
            return None
        return run_dir

    def _copy_to_run_dir(self, ref: JobRef) -> Path:
        """Copy the job's inbox into a fresh folder outside the SyftBox.

        The folder is removed when the copy fails, before the error is raised.
        """
        run_dir = Path(tempfile.mkdtemp(prefix=RUN_DIR_PREFIX)) / RUN_DIR_NAME
        try:
            copy_submission(self.manager.submission_dir(ref), run_dir)
        except BaseException:
            _remove_run_dir(run_dir)
            raise
        return run_dir

    def _return_to_pending(self, ref: JobRef) -> None:
        state = self.manager.read_state(ref)
        state.status = JobStatus.PENDING
        state.approved_by = None
        state.approved_at = None
        state.approval_method = None
        state.review_reason = NO_APPROVED_HASH_REASON
        self.manager.write_state(ref, state)
        print(f" Job {ref.job_name} not run: {NO_APPROVED_HASH_REASON}")

    def _fail_unverified(self, ref: JobRef) -> None:
        self._fail_job(ref, CHANGED_SUBMISSION_REASON)

    def _fail_job(self, ref: JobRef, reason: str) -> None:
        state = self.manager.read_state(ref)
        state.status = JobStatus.FAILED
        state.completed_at = datetime.now(timezone.utc)
        state.review_reason = reason
        self.manager.write_state(ref, state)
        print(f" Job {ref.job_name} not run: {reason}")

    def _load_policies(self, ref: JobRef) -> tuple[bool, JobPolicies | None]:
        """The policies of the api this job calls, from the owner record.

        A job with no api runs with no policies. A job whose api can't be
        loaded fails, and returns (False, None): it never runs without them.
        """
        record = self.manager.read_owner_record(ref)
        if record is None or not record.api_name:
            return True, None
        try:
            if self.api_resolver is None:
                raise PolicyResolutionError("no api resolver is configured")
            return True, self.api_resolver(record.api_name).policies
        except PolicyResolutionError as e:
            self._fail_job(ref, f"api call not run: {e}")
            return False, None

    def _run_approved_job(
        self,
        ref: JobRef,
        run_dir: Path,
        stream_output: bool,
        timeout: int,
        policies: JobPolicies | None = None,
        approved_peers: list[str] | None = None,
    ) -> None:
        job_name = ref.job_name
        self._prepare_outputs_dir(ref, run_dir)

        print(f" Executing job: {job_name}")
        print(f" Running from: {run_dir}")
        self._mark_running(ref)

        try:
            if not self._run_input_policy(ref, run_dir, policies):
                return
            if stream_output:
                returncode = self._execute_job_streaming(ref, timeout, run_dir)
            else:
                returncode = self._execute_job_captured(ref, timeout, run_dir)

            returncode = self._run_output_policy(
                ref, run_dir, policies, returncode, approved_peers
            )

            self._copy_outputs_to_review(ref, run_dir)

            self._finalize(ref, returncode)
            self._report_result(ref, returncode)

        except subprocess.TimeoutExpired:
            print(f" Job {job_name} timed out after {_format_timeout(timeout)}")
            self._finalize(ref, -1)
        except Exception as e:
            print(f" Error executing job {job_name}: {e}")
            self._finalize(ref, -1)

    def _mark_running(self, ref: JobRef) -> None:
        state = self.manager.read_state(ref)
        state.status = JobStatus.RUNNING
        self.manager.write_state(ref, state)

    def _run_input_policy(
        self, ref: JobRef, run_dir: Path, policies: JobPolicies | None
    ) -> bool:
        """Run the input policy. On a deny, fail the job before it runs."""
        if policies is None or policies.input is None:
            return True
        result = self._run_policy(PolicyKind.INPUT, ref, run_dir, policies)
        if result.passed:
            return True
        self._deny(ref, PolicyKind.INPUT, result)
        self._finalize(ref, 1)
        return False

    def _run_output_policy(
        self,
        ref: JobRef,
        run_dir: Path,
        policies: JobPolicies | None,
        returncode: int,
        approved_peers: list[str] | None,
    ) -> int:
        """Run the output policy on the approved read permissions; return the exit code.

        The policy starts from the api's approved read permissions and may change
        them. A deny approves none and fails the job. A failed job skips the
        policy, and the runner then shares nothing.
        """
        if policies is None or policies.output is None or returncode != 0:
            return returncode
        approved = self.manager.read_state(ref).approved_read_permissions
        # The policy edits a temp copy of these (read_permissions.json), not state.yaml.
        result = self._run_policy(PolicyKind.OUTPUT, ref, run_dir, policies, approved)
        if not result.passed:
            self._set_approved(ref, {})
            self._deny(ref, PolicyKind.OUTPUT, result)
            return 1
        # result.read_permissions is that copy as the policy left it.
        allowed = self._allowed_readers(ref, result.read_permissions, approved_peers)
        # Only the submitter and approved peers remain; this writes state.yaml.
        self._set_approved(ref, allowed)
        return returncode

    def _set_approved(self, ref: JobRef, approved: ReadPermissions) -> None:
        state = self.manager.read_state(ref)
        state.approved_read_permissions = approved
        self.manager.write_state(ref, state)

    def _add_approved(self, ref: JobRef, added: ReadPermissions) -> None:
        """Add ``added`` to the approved read permissions; no write if nothing is new."""
        state = self.manager.read_state(ref)
        merged = merge_read_permissions(state.approved_read_permissions, added)
        if merged != state.approved_read_permissions:
            state.approved_read_permissions = merged
            self.manager.write_state(ref, state)

    def _allowed_readers(
        self,
        ref: JobRef,
        permissions: ReadPermissions,
        approved_peers: list[str] | None,
    ) -> ReadPermissions:
        """``permissions`` with only the submitter and approved peers as readers."""
        allowed = set(approved_peers or []) | {ref.ds_email}
        named = {email for emails in permissions.values() for email in emails}
        if named - allowed:
            logger.warning(
                f"Job {ref.job_name}: no read for {sorted(named - allowed)}, "
                f"they are not approved peers"
            )
        kept = {
            f: [e for e in emails if e in allowed] for f, emails in permissions.items()
        }
        return {f: emails for f, emails in kept.items() if emails}

    def _run_policy(
        self,
        kind: PolicyKind,
        ref: JobRef,
        run_dir: Path,
        policies: JobPolicies,
        read_permissions: ReadPermissions | None = None,
    ) -> PolicyResult:
        """Run one policy of the job's api; its full stderr stays with the DO."""
        policy_env = PolicyEnv(
            api_name=policies.api_name,
            submitter=ref.ds_email,
            params=_read_params(run_dir, policies.params_file),
            call_files=_read_call_files(run_dir, policies.call_files),
        )
        if kind == PolicyKind.OUTPUT:
            policy_env.outputs_dir = run_dir / "code" / "outputs"
            policy_env.read_permissions = read_permissions or {}
        print(f" Running the {kind.value} policy of api '{policies.api_name}'")
        state = PolicyState(policies.state_dir)
        result = self.policy_runner.run(policies.script(kind), state, policy_env)
        self._write_artifact(ref, policy_stderr_filename(kind), result.stderr)
        return result

    def _deny(self, ref: JobRef, kind: PolicyKind, result: PolicyResult) -> None:
        """Share the reason of a deny with the submitter."""
        self._write_artifact(ref, POLICY_ERROR_FILENAME, f"{result.error}\n")
        self._get_job_info(ref).share({POLICY_ERROR_FILENAME: [ref.ds_email]})
        print(f" Job {ref.job_name} denied by the {kind.value} policy: {result.error}")

    def _write_artifact(self, ref: JobRef, filename: str, content: str) -> None:
        review_dir = self.manager.review_dir(ref)
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / filename).write_text(content)

    def _report_result(self, ref: JobRef, returncode: int) -> None:
        """Print how the job ended and where its logs are."""
        review_dir = self.manager.review_dir(ref)
        stdout_file = review_dir / "stdout.txt"
        stderr_file = review_dir / "stderr.txt"
        if returncode == 0:
            print(f" Job {ref.job_name} completed successfully")
            print(f" Output written to {stdout_file}")
            return
        print(f" Job {ref.job_name} completed with return code {returncode}")
        print(f" Output written to {stdout_file}")
        try:
            if stderr_file.exists() and stderr_file.stat().st_size > 0:
                print(f" Error output written to {stderr_file}")
        except OSError:
            pass

    def _finalize(self, ref: JobRef, returncode: int) -> None:
        """Write the exit code, and set the state to DONE or FAILED.

        The exit code carries more than DONE or FAILED, so it goes to its own
        file, not to the state that the DS reads.
        """
        review_dir = self.manager.review_dir(ref)
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "returncode.txt").write_text(str(returncode))

        state = self.manager.read_state(ref)
        state.status = JobStatus.DONE if returncode == 0 else JobStatus.FAILED
        state.completed_at = datetime.now(timezone.utc)
        # The exact code is in returncode.txt.
        state.return_code = None
        self.manager.write_state(ref, state)

    def _copy_outputs_to_review(self, ref: JobRef, run_dir: Path) -> None:
        """Copy the outputs from the run copy to review/, with no grant.

        The copy follows symlinks, so review/ holds only real files. In review/,
        a permission file would decide who reads the outputs, so the copy skips
        each one unless the config allows them. macOS and Windows match the
        name without case, so the skip does too.
        """
        run_outputs = run_dir / "code" / "outputs"
        skipped: list[str] = []

        def skip_permission_files(folder: str, names: list[str]) -> set[str]:
            found = {name for name in names if name.casefold() == PERMISSION_FILE_NAME}
            skipped.extend(
                (Path(folder) / name).relative_to(run_outputs).as_posix()
                for name in found
            )
            return found

        shutil.copytree(
            run_outputs,
            self.manager.review_dir(ref) / "outputs",
            ignore=None
            if self.config.allow_output_permission_files
            else skip_permission_files,
        )
        if skipped:
            logger.warning(
                f"Job {ref.job_name}: skipped the permission files in outputs/: "
                f"{', '.join(sorted(skipped))}"
            )

    def _prepare_outputs_dir(self, ref: JobRef, run_dir: Path) -> None:
        """Create outputs/ in the run copy, and delete the files of the last run.

        The job writes its outputs in the run copy, outside the SyftBox, so no
        sync can send them before they are shared. The files of the last run go
        with their grants, or this run inherits them.
        """
        # Create outputs/ inside code/ dir so job scripts can write there (cwd is code/)
        (run_dir / "code" / "outputs").mkdir(parents=True, exist_ok=True)
        self.manager.reset_run(ref)

    def _replace_ds_folder_reads(self) -> None:
        """Replace the read grant on a whole DS review folder with the named files.

        Earlier versions wrote that grant. A job can write any file into its
        review directory, so this runs before any job does.
        """
        review_root = self.config.get_review_dir(self.config.current_user_email)
        if not review_root.is_dir():
            return
        datasite = self.config.syftbox_folder / self.config.current_user_email
        for ds_dir in sorted(p for p in review_root.iterdir() if p.is_dir()):
            replace_ds_folder_read(datasite, ds_dir, ds_dir.name)

    def _get_job_metadata(self, ref: JobRef) -> JobSubmissionMetadata | None:
        try:
            return self.manager.read_submission(ref)
        except Exception:
            return None

    def _get_job_state(self, ref: JobRef) -> JobState:
        try:
            return self.manager.read_state(ref)
        except JobStateNotFoundError:
            return JobState(status=JobStatus.RECEIVED)

    def _get_job_info(self, ref: JobRef) -> JobInfo:
        """Create a JobInfo for a job ref."""
        metadata = self._get_job_metadata(ref)
        if metadata is None:
            raise ValueError(f"Job '{ref.job_name}' not found")

        state = self._get_job_state(ref)
        client = JobClient(config=self.config)
        return JobInfo(
            job_metadata=metadata,
            state=state,
            current_user_email=self.config.current_user_email,
            client=client,
            ref=ref,
        )

    @staticmethod
    def _resolve_skip_jobs(
        approved_jobs: List[JobRef],
        skip_jobs: list[tuple[str, str]] | None,
        skip_job_names: list[str] | None,
    ) -> Set[tuple[str, str]]:
        """The (job_name, ds_email) pairs to skip, accepting the old name-only form.

        A bare name expands to every submitter who used it, which is what
        ``skip_job_names`` did. Bare names in ``skip_jobs`` are read the same
        way, so a caller who passed the old third positional argument keeps the
        behaviour it had.
        """
        pairs: Set[tuple[str, str]] = set()
        names: List[str] = list(skip_job_names or [])
        for entry in skip_jobs or []:
            if isinstance(entry, str):
                names.append(entry)
            else:
                pairs.add((entry[0], entry[1]))

        if names:
            warnings.warn(
                "Skipping jobs by name alone is deprecated: a name is unique "
                "per datasite and submitter, so it also drops other "
                "submitters' jobs of the same name. Pass "
                "skip_jobs=[(job_name, ds_email), ...] instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            wanted = set(names)
            pairs.update(
                (ref.job_name, ref.ds_email)
                for ref in approved_jobs
                if ref.job_name in wanted
            )
        return pairs

    def process_approved_jobs(
        self,
        stream_output: bool = True,
        timeout: int | None = None,
        skip_jobs: list[tuple[str, str]] | None = None,
        share_logs_with_submitter: bool = False,
        share_outputs_with_submitter: bool = False,
        skip_job_names: list[str] | None = None,
        approved_peers: list[str] | None = None,
        share_requested_files: bool = False,
    ) -> None:
        """Process all jobs in approved status.

        The share flags add to each job's approved read permissions before it
        runs, whatever the job requested; once it ran, the runner shares only
        what its state approves. An api call ignores the flags: its api and
        output policy decide.

        Args:
            stream_output: If True (default), stream output in real-time.
            timeout: Timeout in seconds per job. Defaults to the configured value or 600.
            skip_jobs: Optional (job_name, ds_email) pairs to skip. A name alone
                does not identify a job — it is unique per datasite and
                submitter — so skipping by name drops every other job that
                shares it.
            share_logs_with_submitter: If True, approve the logs and the exit
                code for the submitter. False (default) shares only what the
                approval granted. Only the DO reads the rest.
            share_outputs_with_submitter: If True, approve the outputs for the
                submitter.
            skip_job_names: Deprecated. The name-only form of ``skip_jobs``.
            approved_peers: The peers an output policy may give read
                permissions. None allows the submitter only.
            share_requested_files: If True, approve everything each job's
                submitter requested.
        """
        self._replace_ds_folder_reads()
        approved_jobs = self._get_jobs_in_approved()

        if not approved_jobs:
            return

        skip_set = self._resolve_skip_jobs(approved_jobs, skip_jobs, skip_job_names)
        if skip_set:
            approved_jobs = [
                j for j in approved_jobs if (j.job_name, j.ds_email) not in skip_set
            ]

        if not approved_jobs:
            return

        print(f" Found {len(approved_jobs)} job(s) in approved status")

        for ref in approved_jobs:
            self._run_full_job_lifecycle(
                ref,
                stream_output,
                timeout,
                share_logs_with_submitter,
                share_outputs_with_submitter,
                approved_peers,
                share_requested_files,
            )

        if approved_jobs:
            print(f"\n Processed {len(approved_jobs)} job(s)")

    def run_job(
        self,
        job_name: str,
        ds_email: str,
        stream_output: bool = True,
        timeout: int | None = None,
        share_logs_with_submitter: bool = False,
        share_outputs_with_submitter: bool = False,
        approved_peers: list[str] | None = None,
        share_requested_files: bool = False,
    ) -> None:
        """Run one approved job, and share what its state approves.

        Takes the same options as ``process_approved_jobs``. Raises if the job
        is not approved.
        """
        ref = self._find_jobref_from_name(job_name, ds_email)
        status = self._get_job_state(ref).status
        if status != JobStatus.APPROVED:
            raise ValueError(
                f"Job '{job_name}' from {ds_email} is {status.value}, not approved"
            )
        self._replace_ds_folder_reads()
        self._run_full_job_lifecycle(
            ref,
            stream_output,
            timeout,
            share_logs_with_submitter,
            share_outputs_with_submitter,
            approved_peers,
            share_requested_files,
        )

    def _run_full_job_lifecycle(
        self,
        ref: JobRef,
        stream_output: bool,
        timeout: int | None,
        share_logs_with_submitter: bool,
        share_outputs_with_submitter: bool,
        approved_peers: list[str] | None,
        share_requested_files: bool,
    ) -> None:
        """Add the share flags to the approved read permissions, execute the
        job, then share what its state approves if it ran."""
        print(f"\n{'=' * 50}")
        run_permissions = self._run_flag_permissions(
            ref,
            share_logs_with_submitter,
            share_outputs_with_submitter,
            share_requested_files,
        )
        ran = self._execute_job_and_policies(
            ref, stream_output, timeout, approved_peers, run_permissions
        )
        if ran and not self._is_failed_api_call(ref):
            self._share_job_results(ref)
        print(f"{'=' * 50}")

    def _is_failed_api_call(self, ref: JobRef) -> bool:
        """An api call that failed shares nothing: its output policy never ran."""
        if self._get_job_state(ref).status != JobStatus.FAILED:
            return False
        record = self.manager.read_owner_record(ref)
        return record is not None and record.api_name is not None

    def _run_flag_permissions(
        self,
        ref: JobRef,
        share_logs_with_submitter: bool,
        share_outputs_with_submitter: bool,
        share_requested_files: bool,
    ) -> ReadPermissions:
        """The read permissions the share flags approve; none for an api call."""
        if self._is_api_call(ref):
            return {}
        requested = (
            self._get_job_state(ref).requested_read_permissions
            if share_requested_files
            else {}
        )
        return merge_read_permissions(
            _submitter_read_permissions(
                ref.ds_email, share_logs_with_submitter, share_outputs_with_submitter
            ),
            requested,
        )

    def _is_api_call(self, ref: JobRef) -> bool:
        record = self.manager.read_owner_record(ref)
        return bool(record and record.api_name)

    def share_job_results(
        self,
        job_name: str,
        share_logs_with_submitter: bool = False,
        share_outputs_with_submitter: bool = False,
        user: str | None = None,
        share_files: ReadPermissions | None = None,
    ) -> None:
        """Approve more readers of a finished job's files, and share them at once.

        The flags approve the logs and exit code, or the outputs, for the
        submitter; ``share_files`` approves {file: [emails]}. Works on an api
        call whose output policy held the outputs back. Adds to the approved
        read permissions, since a read already given is not taken back.

        Raises:
            ValueError: If the job has not finished.
            TypeError: If share_files is not {file: [emails]}.
        """
        ref = self._find_jobref_from_name(job_name, user)
        status = self._get_job_state(ref).status
        if status not in (JobStatus.DONE, JobStatus.FAILED):
            raise ValueError(f"Job '{job_name}' is {status.value}; run it first.")
        added = merge_read_permissions(
            _submitter_read_permissions(
                ref.ds_email, share_logs_with_submitter, share_outputs_with_submitter
            ),
            validate_read_permissions(share_files),
        )
        self._add_approved(ref, added)
        self._share_job_results(ref)

    def _share_job_results(self, ref: JobRef) -> None:
        """Grant each reader in the job's approved read permissions read on its file."""
        job_info = self._get_job_info(ref)
        job_info.share_read_permissions()
        if job_info.api_name:
            self._update_outputs_withheld_marker(ref, job_info)

    def _update_outputs_withheld_marker(self, ref: JobRef, job_info: JobInfo) -> None:
        """Tell the submitter, with an empty marker, that it can't read the outputs.

        Their job then shows "Outputs not shared with you" instead of no outputs.
        The marker goes once the outputs are approved for them.
        """
        marker = self.manager.review_dir(ref) / OUTPUTS_WITHHELD_FILENAME
        if ref.ds_email in job_info.approved_read_permissions.get(OUTPUTS, []):
            marker.unlink(missing_ok=True)
            return
        self._write_artifact(ref, OUTPUTS_WITHHELD_FILENAME, "")
        job_info.share({OUTPUTS_WITHHELD_FILENAME: [ref.ds_email]})

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
            job_names = ", ".join(
                f"{ref.ds_email}/{ref.job_name}" for ref in self.known_jobs
            )
            print(f" Found {len(self.known_jobs)} existing pending jobs: {job_names}")
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
