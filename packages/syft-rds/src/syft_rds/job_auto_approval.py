"""
Job auto-approval utilities for automatically approving jobs that match specific criteria.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from syft_job.job import JobInfo
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

if TYPE_CHECKING:
    from syft_rds.client import SyftRDSClient


def _get_non_empty_lines(content: str) -> list[str]:
    """Get non-empty lines from content."""
    return [line for line in content.splitlines() if line.strip()]


def _file_content_matches(file_path: Path, expected_content: str) -> bool:
    """
    Check if file content matches expected content (comparing non-empty lines only).

    Args:
        file_path: Path to the file to check
        expected_content: Expected file content

    Returns:
        True if content matches, False otherwise
    """
    try:
        actual = file_path.read_text(encoding="utf-8")
        return _get_non_empty_lines(actual) == _get_non_empty_lines(expected_content)
    except Exception:
        return False


RUN_SCRIPT_PATH = "run.sh"
SUBMISSION_ROOT_FILES = frozenset({RUN_SCRIPT_PATH, "config.yaml"})

# config.yaml carries the job name and the time it was submitted, so its bytes
# are unique to one job. Criteria that pin them match that job and no other.
PER_JOB_FILES = frozenset({"config.yaml"})
CODE_DIR_NAME = "code"


def _get_user_files(job: JobInfo) -> dict[str, Path]:
    """Every file in a job as {path relative to the submission root: abs_path}.

    The root holds `code/`, `run.sh` and `config.yaml`. `run.sh` is the only
    file the runner executes, so it must be in this map. The exact name
    `syft.pub.yaml` is left out, because the permission layer writes it. Any
    other spelling stays in, and is refused as an extra file: an exclusion from
    this inventory is a file nobody reviews, so keep it as narrow as the writer.
    """
    user_files: dict[str, Path] = {}
    root = job.job_submission_path
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and f.name != PERMISSION_FILE_NAME:
                user_files[f.relative_to(root).as_posix()] = f
    return user_files


def validate_criteria_paths(
    required_file_contents: Dict[str, str], required_file_paths: List[str]
) -> None:
    """Refuse criteria written against `<job>/code/`.

    A path is relative to the job submission root, so code sits under `code/`
    and the script the runner executes is `run.sh`. A bare name meant a file
    anywhere under `code/` before, and criteria that keep one approve nothing.
    Say so, rather than rewrite what the caller wrote. Criteria that pin no
    `run.sh` approve nothing either, and they raise here for the same reason:
    a silent no-op tells the owner nothing.
    """
    bare = sorted(
        {
            name
            for name in (*required_file_contents, *required_file_paths)
            if "/" not in name and name not in SUBMISSION_ROOT_FILES
        }
    )
    if bare:
        raise ValueError(
            f"auto-approval criteria name a file relative to the job submission "
            f"root. Write {[f'{CODE_DIR_NAME}/{name}' for name in bare]} instead "
            f"of {bare}."
        )

    per_job = sorted(set(required_file_contents) & PER_JOB_FILES)
    if per_job:
        raise ValueError(
            f"auto-approval criteria cannot pin the content of {per_job}: the "
            f"bytes carry the job name and the time it was submitted, so no "
            f"second job would match. Name them in required_file_paths instead."
        )

    missing_root = sorted(SUBMISSION_ROOT_FILES - set(required_file_paths))
    if missing_root:
        raise ValueError(
            f"required_file_paths must name {missing_root}: every submission "
            f"holds them at its root, so criteria that leave them out match no "
            f"job at all."
        )

    unlisted = sorted(set(required_file_contents) - set(required_file_paths))
    if unlisted:
        raise ValueError(
            f"{unlisted} are pinned by content but missing from "
            f"required_file_paths, which names every file the job may hold. No "
            f"job can hold a file and not hold it, so add them to that list."
        )

    if RUN_SCRIPT_PATH not in required_file_contents:
        raise ValueError(
            f"auto-approval criteria must pin the content of "
            f"'{RUN_SCRIPT_PATH}': it is the file the runner executes, and a "
            f"name in required_file_paths is not enough. Take the script from a "
            f"job you have reviewed, with `job.run_script`."
        )


def job_matches_criteria(
    job: JobInfo,
    required_file_contents: Dict[str, str],
    required_file_paths: List[str],
    allowed_users: Optional[List[str]] = None,
    peers_only: bool = False,
    approved_peers: Optional[List[str]] = None,
) -> bool:
    """
    Check if a job matches all the auto-approval criteria.

    Paths are relative to the job submission root, so code lives under "code/"
    and the script the runner executes is "run.sh". A criteria set that pins no
    content for "run.sh" is refused.

    Args:
        job: JobInfo object to check
        required_file_contents: Dict mapping relative path to expected content
        required_file_paths: List of relative paths that must exist
        allowed_users: Optional list of allowed user emails
        peers_only: If True, only approve jobs from approved peers
        approved_peers: List of approved peer emails (required when peers_only=True)

    Returns:
        True if job matches all criteria, False otherwise

    Raises:
        ValueError: the criteria can approve no job at all. They name a file
            relative to code/ rather than to the submission root, leave run.sh
            or config.yaml out of required_file_paths, pin no content for
            run.sh, pin the content of config.yaml, or pin a path that
            required_file_paths does not name.
    """
    validate_criteria_paths(required_file_contents, required_file_paths)

    # Check status - only process pending jobs
    if job.status != "pending":
        return False

    # Check allowed users filter
    if allowed_users is not None and job.submitted_by not in allowed_users:
        return False

    # Check peers filter
    if peers_only:
        if approved_peers is None:
            return False
        if job.submitted_by not in approved_peers:
            return False

    user_files = _get_user_files(job)

    # Check for all required scripts with exact content match
    for rel_path, expected_content in required_file_contents.items():
        job_file = user_files.get(rel_path)
        if job_file is None or not _file_content_matches(job_file, expected_content):
            return False

    # Check that job contains exactly the required files (no more, no less)
    required_set = set(required_file_paths)
    if set(user_files) != required_set:
        return False

    return True


def auto_approve_and_run_jobs(
    client: SyftRDSClient,
    *,
    required_file_contents: Dict[str, str],
    required_file_paths: List[str],
    allowed_users: Optional[List[str]] = None,
    peers_only: bool = False,
    on_approve: Optional[Callable[[JobInfo], None]] = None,
    verbose: bool = True,
) -> List[JobInfo]:
    """
    Auto-approve and run jobs that match specific criteria.

    This function scans through jobs, approves those that match, and runs them.
    It approves jobs that:
    1. Are in "inbox" status
    2. Contain all specified files with exact content match, run.sh included
    3. Contain all required files, and no others
    4. (Optional) Were submitted by an allowed user
    5. (Optional) Were submitted by an approved peer

    Every path is relative to the job submission root.

    Args:
        client: SyftRDSClient instance
        required_file_contents: Dict mapping relative path to expected content.
                         Content is compared after stripping trailing whitespace.
                         Must pin "run.sh", the file the runner executes; take
                         its content from a job you have reviewed.
                         Example: {"code/main.py": "print('hello')"}
        required_file_paths: List of relative paths that must exist in the job.
                           Example: ["code/main.py", "run.sh", "config.yaml"]
        allowed_users: Optional list of email addresses allowed to submit jobs.
                      If None, any user is allowed (subject to peers_only).
        peers_only: If True, only approve jobs from approved peers.
        on_approve: Optional callback invoked for each approved job.
        verbose: If True, print status messages during approval.

    Returns:
        List of JobInfo objects that were approved.

    Example:
        >>> from syft_rds import login_do
        >>> client = login_do(email="me@example.com", ...)
        >>> approved = auto_approve_and_run_jobs(
        ...     client,
        ...     required_file_contents={
        ...         "code/main.py": EXPECTED_SCRIPT,
        ...         "run.sh": EXPECTED_RUN_SCRIPT,
        ...     },
        ...     required_file_paths=[
        ...         "code/main.py",
        ...         "code/params.json",
        ...         "run.sh",
        ...         "config.yaml",
        ...     ],
        ...     peers_only=True,
        ... )
    """
    validate_criteria_paths(required_file_contents, required_file_paths)

    # Get approved peers if filtering by peers
    approved_peers = None
    if peers_only:
        client.load_peers()
        approved_peers = [p.email for p in client.peer_manager.approved_peers]

    approved_jobs = []
    jobs = client.jobs

    for job in jobs:
        if job_matches_criteria(
            job,
            required_file_contents=required_file_contents,
            required_file_paths=required_file_paths,
            allowed_users=allowed_users,
            peers_only=peers_only,
            approved_peers=approved_peers,
        ):
            try:
                job.approve(approval_method="auto")
                approved_jobs.append(job)

                if on_approve is not None:
                    on_approve(job)

            except Exception as e:
                if verbose:
                    print(f"Failed to approve job '{job.name}': {e}")

    # Run all approved jobs
    if approved_jobs:
        client.process_approved_jobs(stream_output=verbose)

    return approved_jobs
