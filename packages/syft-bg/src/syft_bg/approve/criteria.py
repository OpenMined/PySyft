"""Criteria matching for job approval."""

import hashlib
from pathlib import Path

from syft_job.job import JobInfo
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from pydantic import BaseModel

from syft_bg.approve.config import AutoApprovalObj


class AutoApprovalValidationResult(BaseModel):
    match: bool
    reason: str


RUN_SCRIPT_PATH = "run.sh"


def _compute_file_hash(file_path: Path) -> str | None:
    """Compute SHA256 hash of a file's content."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        return None


def _hash_matches(actual_hash: str, expected_hash: str) -> bool:
    """Compare a file digest against a configured one.

    The compare is over the whole digest. A prefix compare let an empty or
    truncated configured hash match any file.
    """
    if expected_hash.startswith("sha256:"):
        expected = expected_hash[7:]
    else:
        expected = expected_hash
    return bool(expected) and actual_hash == expected


def _content_matches(job_file: Path, stored_path: str) -> bool:
    """Compare file content against stored copy."""
    try:
        stored = Path(stored_path).expanduser()
        if not stored.exists():
            return False
        return job_file.read_text(encoding="utf-8") == stored.read_text(
            encoding="utf-8"
        )
    except Exception:
        return False


def get_job_submission_files(job: JobInfo) -> dict[str, Path]:
    """Every file in a job as {path relative to the submission root: abs_path}.

    The root holds `code/`, `run.sh` and `config.yaml`. `run.sh` is the only
    file the runner executes, so it must be in this set. The exact name
    `syft.pub.yaml` is left out, because the permission layer writes it. Any
    other spelling stays in, and is refused as an extra file: an exclusion from
    this inventory is a file nobody reviews, so keep it as narrow as the writer.
    """
    root = job.job_submission_path
    files: dict[str, Path] = {}
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and f.name != PERMISSION_FILE_NAME:
                files[f.relative_to(root).as_posix()] = f
    return files


def _validate_job_against_object(
    job: JobInfo, obj: AutoApprovalObj
) -> AutoApprovalValidationResult:
    """Validate a job against a single AutoApprovalObj.

    Two-step validation for each content-matched file:
    1. Hash must match a file entry in the object
    2. Content must match the stored copy

    All other files must be in the file_paths allowlist.

    Returns:
        AutoApprovalValidationResult(match=True, reason="ok") if all files pass
        AutoApprovalValidationResult(match=False, reason=...) if any file fails
    """
    # Build lookup: relative_path → FileEntry (content-matched files)
    expected_contents = {entry.relative_path: entry for entry in obj.file_contents}
    expected_names = set(obj.file_paths)
    all_expected_paths = set(expected_contents.keys()) | expected_names
    job_files = get_job_submission_files(job)

    # run.sh is the only file that executes, so a name in the allowlist is not
    # enough for it. An object written before run.sh was validated pins nothing
    # for it and is refused here.
    if RUN_SCRIPT_PATH not in expected_contents:
        return AutoApprovalValidationResult(
            match=False,
            reason=(
                f"approval object pins no content for {RUN_SCRIPT_PATH}, the file "
                f"the runner executes. Recreate it from a job you have reviewed "
                f"with syft_bg.auto_approve_job(job)."
            ),
        )

    # Job files must exactly match the approval object's expected files
    actual_paths = set(job_files.keys())
    if actual_paths != all_expected_paths:
        missing = all_expected_paths - actual_paths
        extra = actual_paths - all_expected_paths
        parts = []
        if missing:
            parts.append(f"missing files: {missing}")
        if extra:
            parts.append(f"extra files: {extra}")
        return AutoApprovalValidationResult(
            match=False,
            reason=f"job files do not match expected filenames: {'; '.join(parts)}",
        )

    for rel_path, file_entry in expected_contents.items():
        job_file = job_files.get(rel_path)
        expected_hash = file_entry.hash
        expected_path = file_entry.path
        submitted_hash = _compute_file_hash(job_file)
        if submitted_hash is None:
            return AutoApprovalValidationResult(
                match=False, reason=f"could not read file: {rel_path}"
            )
        if not _hash_matches(submitted_hash, expected_hash):
            return AutoApprovalValidationResult(
                match=False,
                reason=f"file hash mismatch for {rel_path}: expected {expected_hash}, got sha256:{submitted_hash}",
            )
        if not _content_matches(job_file, expected_path):
            return AutoApprovalValidationResult(
                match=False,
                reason=f"file content mismatch for {rel_path} against stored copy",
            )

    return AutoApprovalValidationResult(match=True, reason="ok")
