import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import psutil

from .client import JobClient
from .job import JobInfo
from . import __version__
from .config import SyftJobConfig
from .job_storage import JobRef, JobStorage, JobStateNotFoundError
from .models import JobState, JobStatus, JobSubmissionMetadata

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

# Sandbox mode for job execution. Job code is untrusted, but this runner also
# runs on data owners' own machines, where sandboxing is neither expected nor
# always possible -- hence "off" by default. The enclave opts in explicitly.
#
#   off     -- execute the job directly (previous behaviour)
#   on      -- sandbox when supported, warn and continue when not
#   require -- sandbox, or refuse to run the job at all
SANDBOX_ENV_VAR = "SYFT_JOB_SANDBOX"
SANDBOX_UID_ENV_VAR = "SYFT_JOB_SANDBOX_UID"
SANDBOX_GID_ENV_VAR = "SYFT_JOB_SANDBOX_GID"
_SANDBOX_MODES = ("off", "on", "require")

# Environment handed to a sandboxed job. The unsandboxed path still inherits the
# full environment; under the sandbox we pass only what a job legitimately needs,
# so bootstrap secrets in the runner's environment are not exposed to job code.
_SANDBOX_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TZ",
    "TMPDIR",
    "LD_LIBRARY_PATH",  # GPU deploys need the NVIDIA driver libs
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "UV_SYSTEM_PYTHON",
    "UV_CACHE_DIR",
)


class SandboxUnavailableError(RuntimeError):
    """Sandbox required by configuration but not applicable here."""


def get_sandbox_mode() -> str:
    """Job sandbox mode from the environment. Defaults to ``off``."""
    mode = os.environ.get(SANDBOX_ENV_VAR, "off").strip().lower()
    if mode not in _SANDBOX_MODES:
        raise ValueError(
            f"{SANDBOX_ENV_VAR} must be one of {_SANDBOX_MODES}, got {mode!r}"
        )
    return mode


def _sandbox_ids() -> tuple[int, int]:
    from .sandbox import DEFAULT_GID, DEFAULT_UID

    return (
        int(os.environ.get(SANDBOX_UID_ENV_VAR, DEFAULT_UID)),
        int(os.environ.get(SANDBOX_GID_ENV_VAR, DEFAULT_GID)),
    )


def _wrap_in_sandbox(command: List[str]) -> List[str]:
    """Prefix ``command`` with the sandbox wrapper.

    The wrapper is invoked by file path rather than as ``-m syft_job.sandbox``
    so that installing the lockdown does not depend on the ``syft_job`` package
    (and its dependencies) importing successfully inside the job environment.
    """
    from . import sandbox as _sandbox

    uid, gid = _sandbox_ids()
    # "on" is explicitly best-effort; "require" must refuse a partial lockdown
    # (e.g. running as a non-root user, where privileges cannot be dropped).
    best_effort = ["--best-effort"] if get_sandbox_mode() == "on" else []
    return [
        sys.executable,
        os.fspath(Path(_sandbox.__file__).resolve()),
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        *best_effort,
        "--",
        *command,
    ]


def _sandbox_available_or_raise(mode: str) -> bool:
    """Whether to sandbox. Raises in ``require`` mode if we cannot."""
    from . import sandbox as _sandbox

    supported, reason = _sandbox.is_supported()
    if supported:
        return True
    if mode == "require":
        raise SandboxUnavailableError(
            f"{SANDBOX_ENV_VAR}=require but the sandbox cannot be applied: {reason}"
        )
    print(
        f" WARNING: {SANDBOX_ENV_VAR}=on but sandbox unavailable ({reason}); "
        f"running job WITHOUT network isolation"
    )
    return False


def build_job_command(run_script: Path) -> List[str]:
    """Command used to launch a job, wrapped in the sandbox when enabled.

    This sandboxes ``run.sh`` wholesale. It is correct only for submissions
    whose script does no dependency installation -- installers need local
    sockets, which the sandbox denies. Python jobs go through
    :func:`build_two_phase_command` instead.
    """
    direct = ["bash", str(run_script)]
    mode = get_sandbox_mode()
    if mode == "off" or not _sandbox_available_or_raise(mode):
        return direct
    return _wrap_in_sandbox(direct)


# Python version used for the job virtualenv. Mirrors
# ``syft_job.client.RUN_SCRIPT_PYTHON_VERSION``, which generates the equivalent
# run.sh for the unsandboxed path.
JOB_PYTHON_VERSION = "3.12"


class PhaseAError(RuntimeError):
    """Dependency installation failed before the job could be sandboxed."""


def _is_submitter_code(spec: str) -> bool:
    """Whether a dependency string would fetch and build submitter-chosen code.

    Local paths and VCS URLs execute their own build scripts on install, so
    under sandboxing they are refused rather than run: phase A has the network
    and runs unsandboxed, which is precisely the position an attacker wants.
    """
    s = spec.strip().lower()
    if any(s.startswith(p) for p in ("git+", "hg+", "svn+", "bzr+")):
        return True
    if s.startswith((".", "/", "file://")):
        return True
    return " @ " in s or s.startswith("-e ")


def install_dependencies(
    submission_dir: Path, dependencies: List[str], timeout: int = 900
) -> Path:
    """Phase A: build the job's virtualenv with the network available.

    Runs *before* the sandbox is applied, so it must not execute code the
    submitter chose. Two rules enforce that:

    * ``syft-client`` is installed from *this* runner's own install source --
      part of the attested enclave image -- not from whatever the submission
      declared. It may be a local path, so it is installed without the
      wheels-only restriction.
    * every submitter-declared dependency is installed ``--only-binary=:all:``,
      because building a source distribution runs its build backend. Declared
      dependencies that are local paths or VCS URLs are refused outright.

    Returns the interpreter to use for phase B.
    """
    from .install_source import get_syft_client_install_source

    code_dir = submission_dir / "code"
    venv_dir = code_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"

    rejected = [d for d in dependencies if _is_submitter_code(d)]
    declared = [d for d in dependencies if not _is_submitter_code(d)]
    if rejected:
        print(
            f" Sandbox: ignoring {len(rejected)} dependency spec(s) that would build "
            f"submitter-supplied code: {', '.join(rejected)}"
        )

    steps: List[List[str]] = [
        ["uv", "venv", "--python", JOB_PYTHON_VERSION, str(venv_dir)],
        # Trusted: comes from the enclave image, not the submission.
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            get_syft_client_install_source(),
        ],
    ]
    if declared:
        steps.append(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "--only-binary=:all:",
                *declared,
            ]
        )

    for step in steps:
        result = subprocess.run(
            step, cwd=code_dir, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise PhaseAError(
                f"{' '.join(step[:3])} failed ({result.returncode}):\n"
                f"{result.stderr.strip()[-2000:]}"
            )

    return venv_python


def build_two_phase_command(
    submission_dir: Path,
    metadata: JobSubmissionMetadata,
    run_script: Path,
    timeout: int = 900,
) -> List[str]:
    """Install dependencies unsandboxed, then return a sandboxed run command.

    The submitted ``run.sh`` is deliberately not executed for python jobs when
    sandboxing: it interleaves installation and execution in one script, so it
    cannot be split, and its contents are chosen by the submitter. The two
    phases are rebuilt from the declared ``entrypoint`` and ``dependencies``
    instead.
    """
    mode = get_sandbox_mode()
    if mode == "off" or not _sandbox_available_or_raise(mode):
        return ["bash", str(run_script)]

    # Only python submissions carry the metadata needed to split the phases.
    if metadata.type != "python" or not metadata.entrypoint:
        return _wrap_in_sandbox(["bash", str(run_script)])

    python = install_dependencies(
        submission_dir, list(metadata.dependencies or []), timeout=timeout
    )
    return _wrap_in_sandbox(
        ["bash", "-c", f'cd code && exec "$0" "$1"', str(python), metadata.entrypoint]
    )


def build_job_env(config_folder: str, email: str) -> dict:
    """Environment for a job subprocess."""
    if get_sandbox_mode() == "off":
        env = os.environ.copy()
    else:
        env = {
            k: v for k, v in os.environ.items() if k in _SANDBOX_ENV_ALLOWLIST
        }
    env["SYFTBOX_FOLDER"] = config_folder
    env["SYFTBOX_EMAIL"] = email
    env[IS_IN_JOB_ENV_VAR] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    return env


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

    def _build_command(
        self, ref: JobRef, submission_dir: Path, run_script: Path, timeout: int
    ) -> List[str]:
        """Command to launch this job, sandboxed and phase-split when enabled."""
        if get_sandbox_mode() == "off":
            return build_job_command(run_script)
        metadata = self._get_job_metadata(ref)
        if metadata is None:
            return build_job_command(run_script)
        return build_two_phase_command(
            submission_dir, metadata, run_script, timeout=timeout
        )

    def _execute_job_streaming(self, ref: JobRef, timeout: int) -> int:
        """Execute job with real-time streaming output.

        Reads run.sh from inbox/, writes stdout/stderr to review/.
        """
        submission_dir = self.manager.submission_dir(ref)
        review_dir = self.manager.review_dir(ref)
        run_script = submission_dir / "run.sh"
        job_name = ref.job_name

        # Log prefix for streaming output
        log_prefix = f"[{self.config.current_user_email}][{job_name}]"

        # Make run.sh executable
        os.chmod(run_script, 0o755)

        # Prepare environment variables
        env = build_job_env(
            self.config.syftbox_folder_path_str, self.config.current_user_email
        )
        command = self._build_command(ref, submission_dir, run_script, timeout)

        # stdout/stderr go to review/
        stdout_file = review_dir / "stdout.txt"
        stderr_file = review_dir / "stderr.txt"

        import selectors

        with (
            open(stdout_file, "w") as stdout_f,
            open(stderr_file, "w") as stderr_f,
        ):
            process = subprocess.Popen(
                command,
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

    def _execute_job_captured(self, ref: JobRef, timeout: int) -> int:
        """Execute job with captured output (non-streaming).

        Reads run.sh from inbox/, writes stdout/stderr to review/.
        """
        submission_dir = self.manager.submission_dir(ref)
        review_dir = self.manager.review_dir(ref)
        run_script = submission_dir / "run.sh"
        job_name = ref.job_name

        # Make run.sh executable
        os.chmod(run_script, 0o755)

        # Prepare environment variables
        env = build_job_env(
            self.config.syftbox_folder_path_str, self.config.current_user_email
        )

        process = subprocess.Popen(
            self._build_command(ref, submission_dir, run_script, timeout),
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
        ref: JobRef,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> bool:
        """
        Execute run.sh for an approved job.

        Reads run.sh from inbox/, writes all output to review/.

        Args:
            ref: Ref of the job to execute.
            stream_output: If True (default), stream output in real-time.
            timeout: Timeout in seconds. Defaults to 300 (5 minutes).

        Returns:
            bool: True if execution was successful, False otherwise
        """
        if timeout is None:
            timeout = get_job_timeout_seconds()

        job_name = ref.job_name
        submission_dir = self.manager.submission_dir(ref)
        review_dir = self.manager.review_dir(ref)
        run_script = submission_dir / "run.sh"

        if not run_script.exists():
            print(f" No run.sh found in {job_name}")
            return False

        self._prepare_outputs_dir(ref)

        print(f" Executing job: {job_name}")
        print(f" Inbox: {submission_dir}")

        # Update state to RUNNING
        state = self.manager.read_state(ref)
        state.status = JobStatus.RUNNING
        self.manager.write_state(ref, state)

        try:
            if stream_output:
                returncode = self._execute_job_streaming(ref, timeout)
            else:
                returncode = self._execute_job_captured(ref, timeout)

            # Move outputs from inbox/ to review/
            self._move_outputs_to_review(submission_dir, review_dir)

            # Write return code to review/
            returncode_file = review_dir / "returncode.txt"
            with open(returncode_file, "w") as f:
                f.write(str(returncode))

            # Update state to DONE or FAILED
            self._set_finalized_job_state(ref, returncode)

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
            self._set_finalized_job_state(ref, -1)
            return False
        except Exception as e:
            print(f" Error executing job {job_name}: {e}")
            self._set_finalized_job_state(ref, -1)
            return False

    def _set_finalized_job_state(self, ref: JobRef, returncode: int) -> None:
        state = self.manager.read_state(ref)
        state.status = JobStatus.DONE if returncode == 0 else JobStatus.FAILED
        state.completed_at = datetime.now(timezone.utc)
        state.return_code = returncode
        self.manager.write_state(ref, state)

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

    def _prepare_outputs_dir(self, ref: JobRef) -> None:
        """Clear and recreate outputs dir in both inbox/ (for job cwd) and review/ (for final results)."""
        # Create outputs/ inside code/ dir so job scripts can write there (cwd is code/)
        submission_dir = self.manager.submission_dir(ref)
        inbox_outputs = submission_dir / "code" / "outputs"
        inbox_outputs.mkdir(parents=True, exist_ok=True)

        # Create outputs/ in review dir with owner-only read permissions
        review_dir = self.manager.review_dir(ref)
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
            approved_jobs = [j for j in approved_jobs if j.job_name not in skip_set]

        if not approved_jobs:
            return

        print(f" Found {len(approved_jobs)} job(s) in approved status")

        for ref in approved_jobs:
            print(f"\n{'=' * 50}")
            self._execute_job(ref, stream_output=stream_output, timeout=timeout)
            self._share_job_results(
                ref, share_outputs_with_submitter, share_logs_with_submitter
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
        self._share_job_results(
            self._find_jobref_from_name(job_name, user), share_outputs, share_logs
        )

    def _share_job_results(
        self, ref: JobRef, share_outputs: bool, share_logs: bool
    ) -> None:
        if not share_outputs and not share_logs:
            return
        job_info = self._get_job_info(ref)
        if share_outputs:
            job_info.share_outputs([ref.ds_email])
        if share_logs:
            job_info.share_logs([ref.ds_email])

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
