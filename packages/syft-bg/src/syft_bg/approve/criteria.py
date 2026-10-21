"""Criteria matching for job approval.

The match itself lives in syft_rds.apis.matching, so the job runner can check
it again before an api's policies run.
"""

from pathlib import Path

from pydantic import BaseModel
from syft_job.job import JobInfo
from syft_rds.apis.matching import (
    RUN_SCRIPT_PATH,
    compute_file_hash,
    content_matches,
    hash_matches,
    match_submission,
    submission_files,
)

from syft_bg.approve.config import AutoApprovalObj

__all__ = [
    "RUN_SCRIPT_PATH",
    "AutoApprovalValidationResult",
    "get_job_submission_files",
]

# The names these helpers had before they moved.
_compute_file_hash = compute_file_hash
_hash_matches = hash_matches
_content_matches = content_matches


class AutoApprovalValidationResult(BaseModel):
    match: bool
    reason: str
    # The api that matched; its policies run around the job.
    api_name: str | None = None


def get_job_submission_files(job: JobInfo) -> dict[str, Path]:
    """Every file in a job as {path relative to the submission root: abs_path}."""
    return submission_files(job.job_submission_path)


def _validate_job_against_object(
    job: JobInfo, obj: AutoApprovalObj
) -> AutoApprovalValidationResult:
    """Validate a job against a single AutoApprovalObj."""
    match, reason = match_submission(job.job_submission_path, obj)
    return AutoApprovalValidationResult(match=match, reason=reason)
