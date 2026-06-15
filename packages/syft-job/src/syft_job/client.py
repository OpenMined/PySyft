import os
import shutil
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from syft_perms.syftperm_context import SyftPermContext
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from .config import SyftJobConfig
from .install_source import get_syft_client_install_source
from .job import JobInfo, JobsList
from .models.config import JobSubmissionMetadata
from .models.state import JobState, JobStatus

# Python version used when creating virtual environments for job execution
RUN_SCRIPT_PYTHON_VERSION = "3.12"

# Strict schema: only these entries are allowed in a job submission
VALID_SUBMISSION_ENTRIES = {"code", "run.sh", "config.yaml"}


class BaseJobClient(ABC):
    """Interface for job clients."""

    config: SyftJobConfig
    current_user_email: str

    @abstractmethod
    def submit_python_job(
        self,
        user: str,
        code_path: str,
        job_name: Optional[str] = "",
        **kwargs,
    ) -> Path: ...

    @abstractmethod
    def submit_bash_job(self, user: str, script: str, job_name: str = "") -> Path: ...

    @abstractmethod
    def setup_ds_job_folder_as_do(self, ds_email: str) -> Path: ...

    @abstractmethod
    def scan_inbox(self) -> None: ...

    @property
    @abstractmethod
    def jobs(self) -> JobsList: ...


class JobClient(BaseJobClient):
    """Client for submitting jobs to SyftBox."""

    def __init__(
        self, config: SyftJobConfig, target_datasite_owner_email: Optional[str] = None
    ):
        """Initialize JobClient with configuration and optional user email for job views."""
        self.config = config
        self.current_user_email = (
            config.current_user_email
        )  # From SyftBox folder (for "submitted_by")
        self.target_datasite_owner_email = (
            target_datasite_owner_email or config.current_user_email
        )  # The email of the datasite owner of the jobs
        self.has_do_role = config.has_do_role
        self.peer_install_sources: dict[
            str, str
        ] = {}  # do_email -> syft-client install source (local path, git URL, or "syft-client==X.Y.Z") advertised by that DO in their VersionInfo

        # Validate that user_email exists in SyftBox root
        self._validate_user_email()

    @classmethod
    def from_config(cls, config: SyftJobConfig) -> "JobClient":
        return cls(config, config.current_user_email)

    def _validate_user_email(self) -> None:
        """Validate that the user_email directory exists in SyftBox root."""
        user_dir = self.config.get_user_dir(self.target_datasite_owner_email)
        if not user_dir.exists():
            # Create user directory if it doesn't exist
            user_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created user directory: {user_dir}")

    def _ensure_my_submission_directory_exists(
        self, target_datasite_owner_email: str
    ) -> None:
        """Ensure inbox directory structure exists for submitting jobs."""
        my_submission_dir = self.config._get_job_submission_dir_for_me(
            target_datasite_owner_email
        )
        my_submission_dir.mkdir(parents=True, exist_ok=True)

    def setup_ds_job_folder_as_do(self, ds_email: str) -> Path:
        """Create inbox and review subdirectories for a DS with appropriate permissions.

        Called when DO approves a peer — creates the folder structure that allows
        the DS to submit jobs.

        Args:
            ds_email: Email of the data scientist to create the folder for.

        Returns:
            Path to the created DS inbox folder.
        """
        datasite = self.config.syftbox_folder / self.current_user_email

        # Create inbox folder for DS with write access
        ds_inbox_dir = (
            self.config.get_all_submissions_dir(self.current_user_email) / ds_email
        )
        ds_inbox_dir.mkdir(parents=True, exist_ok=True)
        ctx = SyftPermContext(datasite=datasite)
        inbox_rel_dir = ds_inbox_dir.relative_to(datasite)
        ctx.open(inbox_rel_dir).grant_write_access(ds_email)

        # Create review folder for DS with read access
        ds_review_dir = self.config.get_review_dir(self.current_user_email) / ds_email
        ds_review_dir.mkdir(parents=True, exist_ok=True)
        review_rel_dir = ds_review_dir.relative_to(datasite)
        ctx.open(review_rel_dir).grant_read_access(ds_email)

        return ds_inbox_dir

    # ──────────────────────────────────────────────
    # Submission (DS side)
    # ──────────────────────────────────────────────

    def submit_bash_job(self, user: str, script: str, job_name: str = "") -> Path:
        """
        Submit a bash job for a user.

        Args:
            user: Email address of the datasite owner to submit job to
            script: Bash script content to execute
            job_name: Name of the job (directory name). If empty, auto-generated.

        Returns:
            Path to the created job directory in inbox/

        Raises:
            FileExistsError: If job with same name already exists
        """
        submitting_to_email = user
        # Generate default job name if not provided
        if not job_name.strip():
            from uuid import uuid4

            random_id = str(uuid4())[0:8]
            job_name = f"Job - {random_id}"

        # Ensure user directory exists (create if it doesn't)
        user_dir = self.config.get_user_dir(user)
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created user directory: {user_dir}")

        # Ensure inbox directory structure exists
        self._ensure_my_submission_directory_exists(submitting_to_email)

        # Create job directory under inbox/<ds_email>/<job_name>/
        job_dir = self.config.get_job_submission_dir(
            submitting_to_email, self.config.current_user_email, job_name
        )

        if job_dir.exists():
            raise FileExistsError(
                f"Job '{job_name}' already exists for user '{submitting_to_email}'"
            )

        self._write_bash_script(job_dir, script)

        # Write config.yaml using model
        config = JobSubmissionMetadata(
            name=job_name,
            type="bash",
            submitted_by=self.current_user_email,
            datasite_email=submitting_to_email,
            submitted_at=datetime.now(timezone.utc),
            files=["script.sh"],
        )
        config.save(job_dir / "config.yaml")

        return job_dir

    def _write_bash_script(self, job_dir: Path, script: str) -> None:
        job_dir.mkdir(parents=True)
        # Create code/ directory with the script
        code_dir = job_dir / "code"
        code_dir.mkdir()
        (code_dir / "script.sh").write_text(script)

        # Create run.sh at job root
        run_script_path = job_dir / "run.sh"
        run_script_path.write_text(script)
        os.chmod(run_script_path, 0o755)

    def _detect_entrypoint(self, folder_path: Path) -> Optional[str]:
        """Auto-detect entrypoint for folder submissions.

        Detection priority:
        1. main.py - most common convention
        2. Single .py file at root - if only one exists

        Args:
            folder_path: Path to the folder to search

        Returns:
            Detected entrypoint filename or None if not found
        """
        # Check main.py first (most common convention)
        if (folder_path / "main.py").exists():
            return "main.py"

        # Check for single .py file at root
        py_files = [
            f for f in folder_path.iterdir() if f.is_file() and f.suffix == ".py"
        ]
        if len(py_files) == 1:
            return py_files[0].name

        return None

    def _validate_code_path_and_entrypoint(
        self, code_path: str, entrypoint: Optional[str]
    ) -> Tuple[Path, bool, str]:
        """
        Validate code path and entrypoint for Python job submission.

        Args:
            code_path: Path to Python file or folder
            entrypoint: Entry point file name (auto-detected if not provided)

        Returns:
            Tuple of (resolved_code_path, is_folder_submission, validated_entrypoint)

        Raises:
            FileNotFoundError: If code_path doesn't exist
            ValueError: If validation fails
        """
        code_path_input = code_path  # Keep original for error messages
        resolved_path = Path(code_path).expanduser().resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Code path does not exist: {code_path_input}")

        is_folder_submission = resolved_path.is_dir()

        if is_folder_submission:
            if not entrypoint:
                # Auto-detect entrypoint
                entrypoint = self._detect_entrypoint(resolved_path)
                if not entrypoint:
                    raise ValueError(
                        "Could not auto-detect entrypoint. No main.py or single .py file "
                        "found at folder root. Please specify the entrypoint explicitly."
                    )

            entrypoint_path = resolved_path / entrypoint
            if not entrypoint_path.exists() or not entrypoint_path.is_file():
                raise ValueError(
                    f"Entrypoint file '{entrypoint}' not found in folder: {code_path_input}"
                )

            if entrypoint_path.suffix != ".py":
                raise ValueError(
                    f"Entrypoint file must be a Python file (.py): {entrypoint}"
                )
        else:
            if resolved_path.suffix != ".py":
                raise ValueError(
                    f"Code path must be a Python file (.py): {code_path_input}"
                )
            # Auto-detect entrypoint for file submissions
            entrypoint = resolved_path.name

        return resolved_path, is_folder_submission, entrypoint

    def _resolve_install_source_for(self, do_email: str) -> str:
        """Return the syft-client install source to bake into a job's run.sh.

        Prefers the source advertised by the DO during peer-version exchange
        (stored on `self.peer_install_sources`). Falls back to local detection
        with a loud warning if the DO never advertised one (e.g. older client).
        """
        advertised = self.peer_install_sources.get(do_email)
        if advertised:
            return advertised

        fallback = get_syft_client_install_source()
        warnings.warn(
            f"\n{'=' * 78}\n"
            f"WARNING: No syft-client install source advertised by DO {do_email!r}.\n"
            f"Falling back to local detection on the DS side: {fallback!r}.\n"
            f"The DO may be running an older syft-client; the job may fail to\n"
            f"install dependencies on the DO's machine.\n"
            f"{'=' * 78}",
            stacklevel=3,
        )
        return fallback

    def _generate_python_run_script(
        self,
        entrypoint_path: str,
        dependencies: List[str],
        has_pyproject: bool,
        install_source: str,
    ) -> str:
        """
        Generate bash script for Python job execution.

        Args:
            entrypoint_path: Filename to execute (e.g., "main.py" or "script.py")
            dependencies: List of dependencies to install
            has_pyproject: Whether the code has a pyproject.toml
            install_source: Syft-client install source to use (pip/uv-compatible string).

        Returns:
            Bash script content
        """
        all_dependencies = [install_source] + dependencies

        if has_pyproject:
            # For projects with pyproject.toml, run uv sync inside the code folder
            # entrypoint_path is like "code/main.py", code_folder is "code"
            code_folder = "code"
            # Always install syft_client (and any extra dependencies) after uv sync
            deps_str = " ".join(f'"{dep}"' for dep in all_dependencies)
            install_deps_cmd = f"uv pip install {deps_str}"

            return f"""#!/bin/bash
set -euo pipefail
export UV_SYSTEM_PYTHON=false
cd {code_folder}
uv venv --python {RUN_SCRIPT_PYTHON_VERSION}
source .venv/bin/activate
uv sync --python {RUN_SCRIPT_PYTHON_VERSION}
{install_deps_cmd}
export PYTHONPATH=.:${{PYTHONPATH:-}}
python {entrypoint_path}
"""
        else:
            # entrypoint_path is just the filename (e.g. "main.py")
            # We cd into code/ and run from there so relative paths (like params.json) work
            code_folder = "code"

            deps_str = " ".join(f'"{dep}"' for dep in all_dependencies)
            return f"""#!/bin/bash
set -euo pipefail
export UV_SYSTEM_PYTHON=false
cd {code_folder}
uv venv --python {RUN_SCRIPT_PYTHON_VERSION}
source .venv/bin/activate
uv pip install {deps_str}
export PYTHONPATH=.:${{PYTHONPATH:-}}
python {entrypoint_path}
"""

    def submit_python_job(
        self,
        user: str,
        code_path: str,
        job_name: Optional[str] = "",
        dependencies: Optional[List[str]] = None,
        entrypoint: Optional[str] = None,
    ) -> Path:
        """
        Submit a Python job for a user (supports both files and folders).

        Args:
            user: Email address of the datasite owner to submit job to
            code_path: Path to Python file or folder containing Python code
            job_name: Name of the job (directory name). If empty, auto-generated.
            dependencies: List of Python packages to install
            entrypoint: Entry point file name (auto-detected if not provided)

        Returns:
            Path to the created job directory in inbox/

        Raises:
            FileExistsError: If job with same name already exists
            ValueError: If code_path validation fails
            FileNotFoundError: If code_path doesn't exist
        """
        submitting_to_email = user
        # Generate default job name if not provided
        if not job_name:
            from uuid import uuid4

            random_id = str(uuid4())[0:8]
            job_name = f"Job - {random_id}"

        # Validate code path and entrypoint
        code_path_resolved, is_folder_submission, entrypoint = (
            self._validate_code_path_and_entrypoint(code_path, entrypoint)
        )

        # Ensure user directory exists (create if it doesn't)
        user_dir = self.config.get_user_dir(user)
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created user directory: {user_dir}")

        # Ensure inbox directory structure exists
        self._ensure_my_submission_directory_exists(submitting_to_email)

        # Create job directory under inbox/<ds_email>/<job_name>/
        job_dir = self.config.get_job_submission_dir(
            submitting_to_email, self.config.current_user_email, job_name
        )

        if job_dir.exists():
            raise FileExistsError(f"Job '{job_name}' already exists for user '{user}'")

        job_dir.mkdir(parents=True)

        # Copy code into code/ subdirectory
        code_dest = job_dir / "code"
        if is_folder_submission:
            shutil.copytree(code_path_resolved, code_dest)
            # Entrypoint path is relative to code/
            pyproject_path = code_dest / "pyproject.toml"
        else:
            code_dest.mkdir(parents=True)
            shutil.copy2(code_path_resolved, code_dest / code_path_resolved.name)
            pyproject_path = None

        # Resolve install source: prefer what the DO advertised in their VersionInfo,
        # fall back to local detection with a warning if the DO didn't advertise one.
        install_source = self._resolve_install_source_for(submitting_to_email)

        # Generate bash script for Python execution
        dependencies = dependencies or []
        has_pyproject = pyproject_path is not None and pyproject_path.exists()
        bash_script = self._generate_python_run_script(
            entrypoint, dependencies, has_pyproject, install_source
        )

        # Create run.sh file
        run_script_path = job_dir / "run.sh"
        run_script_path.write_text(bash_script)
        os.chmod(run_script_path, 0o755)

        # Build file manifest of code/ contents
        files = [
            str(p.relative_to(code_dest)) for p in code_dest.rglob("*") if p.is_file()
        ]

        # Compute all_dependencies for config
        all_dependencies = [install_source] + dependencies

        # Write config.yaml using model
        config = JobSubmissionMetadata(
            name=job_name,
            type="python",
            submitted_by=self.current_user_email,
            datasite_email=submitting_to_email,
            submitted_at=datetime.now(timezone.utc),
            entrypoint=entrypoint,
            dependencies=all_dependencies,
            files=files,
            is_folder_submission=is_folder_submission,
            code_path=str(code_path_resolved),
        )
        config.save(job_dir / "config.yaml")

        return job_dir

    # ──────────────────────────────────────────────
    # Validation and reception (DO side)
    # ──────────────────────────────────────────────

    def validate_submission(self, inbox_job_path: Path) -> tuple[bool, str]:
        """Check strict schema: only code/ + run.sh + config.yaml allowed.

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        entries = {
            e.name for e in inbox_job_path.iterdir() if e.name != PERMISSION_FILE_NAME
        }
        if entries != VALID_SUBMISSION_ENTRIES:
            return False, f"Expected {VALID_SUBMISSION_ENTRIES}, got {entries}"
        if not (inbox_job_path / "code").is_dir():
            return False, "'code' must be a directory"
        if not (inbox_job_path / "run.sh").is_file():
            return False, "'run.sh' must be a file"
        if not (inbox_job_path / "config.yaml").is_file():
            return False, "'config.yaml' must be a file"
        return True, ""

    def receive_job(self, ds_email: str, job_name: str) -> JobState:
        """Validate an incoming job and create initial state in review/.

        Called by scan_inbox() when a new job is detected.

        Args:
            ds_email: Email of the data scientist who submitted the job.
            job_name: Name of the job.

        Returns:
            The created JobState.
        """
        submission_path = self.config.get_job_submission_dir(
            self.current_user_email, ds_email, job_name
        )
        review_path = self.config.get_review_job_dir(
            self.current_user_email, ds_email, job_name
        )

        now = datetime.now(timezone.utc)
        valid, reason = self.validate_submission(submission_path)

        if not valid:
            state = JobState(
                status=JobStatus.REJECTED,
                received_at=now,
                rejected_by=self.current_user_email,
                rejected_at=now,
                review_reason=reason,
            )
        else:
            state = JobState(status=JobStatus.PENDING, received_at=now)

        review_path.mkdir(parents=True, exist_ok=True)
        state.save(review_path / "state.yaml")
        return state

    def scan_inbox(self) -> None:
        """Scan inbox/ for new unprocessed jobs and receive them.

        For each job in inbox/ that doesn't have a corresponding state.yaml
        in review/, validates the submission and creates the initial state.
        """
        inbox_dir = self.config.get_all_submissions_dir(self.current_user_email)
        if not inbox_dir.exists():
            return

        for ds_dir in inbox_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for job_dir in ds_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                if not (job_dir / "config.yaml").exists():
                    continue

                # Already processed?
                review_state = (
                    self.config.get_review_job_dir(
                        self.current_user_email, ds_dir.name, job_dir.name
                    )
                    / "state.yaml"
                )
                if review_state.exists():
                    continue

                self.receive_job(ds_dir.name, job_dir.name)

    # ──────────────────────────────────────────────
    # Listing
    # ──────────────────────────────────────────────

    def _get_all_jobs(self) -> List[JobInfo]:
        """Get all jobs by scanning inbox/ and correlating with review/."""
        jobs: list[JobInfo] = []
        syftbox_root = self.config.syftbox_folder

        if not syftbox_root.exists():
            return jobs

        # Scan through all user directories in SyftBox root (peers)
        for datasite_owner_dir in syftbox_root.iterdir():
            if not datasite_owner_dir.is_dir():
                continue

            datasite_owner_email = datasite_owner_dir.name
            inbox_dir = self.config.get_all_submissions_dir(datasite_owner_email)

            if not inbox_dir.exists():
                continue

            # Scan DS subdirectories, then job directories within each
            for ds_dir in inbox_dir.iterdir():
                if not ds_dir.is_dir():
                    continue

                for job_dir in ds_dir.iterdir():
                    if not job_dir.is_dir():
                        continue

                    config_file = job_dir / "config.yaml"
                    if not config_file.exists():
                        continue

                    try:
                        config = JobSubmissionMetadata.load(config_file)

                        # Look up state from review/
                        review_path = self.config.get_review_job_dir(
                            datasite_owner_email, ds_dir.name, job_dir.name
                        )
                        state_file = review_path / "state.yaml"
                        if state_file.exists():
                            state = JobState.load(state_file)
                        else:
                            state = JobState(status=JobStatus.RECEIVED)

                        jobs.append(
                            JobInfo(
                                job_metadata=config,
                                state=state,
                                datasite_owner_email=datasite_owner_email,
                                current_user_email=self.current_user_email,
                                client=self,
                            )
                        )
                    except Exception:
                        continue

        return jobs

    @property
    def jobs(self) -> JobsList:
        """
        Get all jobs from all peer directories as an indexable list grouped by user.

        Returns a JobsList object that can be:
        - Indexed: jobs[0], jobs[1], etc.
        - Iterated: for job in jobs
        - Displayed: print(jobs) shows separate tables for each user
        - HTML display: in Jupyter, shows separate tables for each user with jobs

        Returns:
            JobsList containing all jobs from all peer directories, grouped by user
        """
        # Auto-scan current user's inbox for new submissions before listing
        self.scan_inbox()

        current_jobs = self._get_all_jobs()

        # Sort jobs by recent submissions first (newest first), then by user/status
        def job_sort_key(job):
            # Parse submitted_at timestamp for sorting (most recent first)
            try:
                if job.submitted_at:
                    from datetime import datetime as dt

                    # Parse ISO format timestamp
                    ts = dt.fromisoformat(job.submitted_at.replace("Z", "+00:00"))
                    # Use negative timestamp for reverse chronological order (newest first)
                    time_priority = -ts.timestamp()
                else:
                    # Jobs without submitted_at go to the end
                    time_priority = float("inf")
            except Exception:
                # Invalid timestamps go to the end
                time_priority = float("inf")

            # Secondary sorting: user priority (root first), then user name, then status
            user_priority = (
                0 if job.datasite_owner_email == self.current_user_email else 1
            )
            status_order = {
                "received": 0,
                "pending": 1,
                "approved": 2,
                "running": 3,
                "rejected": 4,
                "done": 5,
                "failed": 6,
            }
            status_priority = status_order.get(job.status, 7)

            return (
                time_priority,
                user_priority,
                job.datasite_owner_email,
                status_priority,
                job.name.lower(),
            )

        sorted_jobs = sorted(current_jobs, key=job_sort_key)
        return JobsList(
            sorted_jobs,
            self.target_datasite_owner_email,
            has_do_role=self.has_do_role,
        )


def get_client(
    syftbox_folder_path: str, email: str, user_email: Optional[str] = None
) -> JobClient:
    """
    Factory function to create a JobClient from SyftBox folder.

    Args:
        syftbox_folder_path: Path to the SyftBox folder
        email: Root user email address (explicit, no inference from folder name)
        user_email: Optional target user email for job views (defaults to root email)

    Returns:
        Configured JobClient instance
    """
    config = SyftJobConfig.from_syftbox_folder(syftbox_folder_path, email)
    return JobClient(config, user_email)
