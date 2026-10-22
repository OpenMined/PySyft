"""A job submission: when all its files arrived, and which exact files were approved.

The submitter can write its inbox folder at any time, and sync delivers a job
in any number of batches. A job counts as received only when its files hash to
the value the submitter declared in config.yaml. The data owner then records
the hash of what it received, later the hash of what it approved, and the
runner executes a fresh copy only after checking it against the approved hash.
The record lives in the owner's staging/, which no submitter can write.
"""

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from pydantic import BaseModel
from syft_permissions import Access, Rule, RuleSet
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

# Strict schema: only these entries are allowed in a job submission.
SUBMISSION_ENTRIES = frozenset({"code", "run.sh", "config.yaml"})

# A request (an api call) holds request.yaml plus the files the call sends;
# which files those may be is the api's to say.
REQUEST_FILE_NAME = "request.yaml"

# The header of a request's config.yaml that names the api it calls.
API_HEADER = "api"

# The entries the submitter's hash covers. config.yaml carries that hash, so it
# cannot be part of it.
CODE_ENTRIES = frozenset({"code", "run.sh"})

# Names that sync never carries, mirrored from syft.sync.utils.path_filters,
# which syft-job cannot import. A test in tests/unit keeps the two equal.
SYNC_EXCLUDED_NAMES = frozenset(
    {
        ".venv",
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".DS_Store",
    }
)

# config.yaml header where the submitter declares code_hash() of its job.
SUBMISSION_HASH_HEADER = "submission_hash"

# The data owner's record of a job, in staging/. Releases before the rename
# wrote it as submission.json; JobStorage moves that file on first access.
OWNER_RECORD_FILENAME = "owner_record.json"
LEGACY_OWNER_RECORD_FILENAME = "submission.json"


class InvalidSubmissionError(ValueError):
    """The submission holds a symlink, or not the expected entries."""


class OwnerJobRecord(BaseModel):
    """The data owner's own record of a job: what it received and approved.

    Only the data owner writes it, so the submitter can't change it.
    """

    received_hash: str
    received_at: datetime
    approved_hash: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_method: Optional[str] = None
    # Set when the DO turned an api call (a request) into this job. The runner
    # runs the job with that api's policies. Only the DO writes it.
    api_name: Optional[str] = None


def _is_skipped(relative: Path) -> bool:
    """Permission files, in any case, and names that sync never carries."""
    if relative.name.casefold() == PERMISSION_FILE_NAME.casefold():
        return True
    return any(part in SYNC_EXCLUDED_NAMES for part in relative.parts)


def _as_sent(path: Path) -> bytes:
    """The bytes the data owner ends up with for ``path``.

    The submitter's syncer reads a file as text and the owner writes it back
    as UTF-8, so line endings and the local encoding do not survive.
    """
    try:
        with open(path, "r") as f:
            return f.read().encode("utf-8")
    except UnicodeDecodeError:
        return path.read_bytes()


def _digest(
    submission_dir: Path,
    entries: Optional[frozenset[str]],
    as_sent: bool,
    skip: frozenset[str] = frozenset(),
) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in submission_dir.rglob("*")
        if path.is_file()
        and not _is_skipped(relative := path.relative_to(submission_dir))
        and (entries is None or relative.parts[0] in entries)
        and relative.as_posix() not in skip
    )
    for path in files:
        content = _as_sent(path) if as_sent else path.read_bytes()
        digest.update(path.relative_to(submission_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def code_hash(submission_dir: Path, as_sent: bool = False) -> str:
    """SHA-256 over run.sh and code/: the hash the submitter declares.

    ``as_sent`` hashes the bytes as the data owner will receive them; the
    submitter passes it, the owner hashes its files as they are on disk.
    """
    return _digest(submission_dir, CODE_ENTRIES, as_sent)


def request_hash(request_dir: Path, as_sent: bool = False) -> str:
    """SHA-256 over every file of a request but request.yaml, which carries it."""
    return _digest(request_dir, None, as_sent, skip=frozenset({REQUEST_FILE_NAME}))


def submission_hash(submission_dir: Path) -> str:
    """SHA-256 over every file of a submission: code/, run.sh and config.yaml.

    config.yaml carries the job name, the submission time and the datasets, so
    one digest pins the job, its code and its dataset set. Permission files and
    names that sync never carries are left out.
    """
    return _digest(submission_dir, None, as_sent=False)


def validate_submission(submission_dir: Path) -> tuple[bool, str]:
    """Check the strict schema: only code/ + run.sh + config.yaml, and no symlinks.

    A symlink could point at any file on the data owner's machine.

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    entries = {
        e.name for e in submission_dir.iterdir() if e.name != PERMISSION_FILE_NAME
    }
    if entries != SUBMISSION_ENTRIES:
        return False, f"Expected {set(SUBMISSION_ENTRIES)}, got {entries}"
    if not (submission_dir / "code").is_dir():
        return False, "'code' must be a directory"
    if not (submission_dir / "run.sh").is_file():
        return False, "'run.sh' must be a file"
    if not (submission_dir / "config.yaml").is_file():
        return False, "'config.yaml' must be a file"
    return _no_symlinks(submission_dir)


def validate_request(request_dir: Path) -> tuple[bool, str]:
    """Check a request has request.yaml and no symlinks.

    Which other files it may hold is the api's to say, checked against it.
    """
    if not (request_dir / REQUEST_FILE_NAME).is_file():
        return False, f"'{REQUEST_FILE_NAME}' must be a file"
    return _no_symlinks(request_dir)


def _no_symlinks(folder: Path) -> tuple[bool, str]:
    for path in folder.rglob("*"):
        if path.is_symlink():
            return False, f"'{path.relative_to(folder).as_posix()}' is a symlink"
    return True, ""


def has_listed_files(submission_dir: Path, files: Iterable[str]) -> bool:
    """Whether run.sh and every code/ file in the manifest are present.

    The completeness check for a submitter that declares no hash.
    """
    if not (submission_dir / "run.sh").is_file():
        return False
    return all(
        (submission_dir / "code" / name).is_file()
        for name in files
        if not _is_skipped(Path(name))
    )


def copy_submission(source: Path, target: Path) -> None:
    """Copy a submission file by file, skipping what ``submission_hash`` skips.

    Raises:
        InvalidSubmissionError: a symlink appeared in the source.
    """
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if _is_skipped(relative):
            continue
        if path.is_symlink():
            raise InvalidSubmissionError(f"'{relative.as_posix()}' is a symlink")
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def read_owner_record(path: Path) -> Optional[OwnerJobRecord]:
    """The record at ``path``, or None when absent or unreadable."""
    try:
        return OwnerJobRecord.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return None


def claim_owner_record(path: Path, record: OwnerJobRecord) -> bool:
    """Create the record only if there is none; True when this call created it.

    Two data owner processes can receive the same job at once. Only the one
    that creates the record finishes the receipt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(record.model_dump_json())
        f.flush()
        os.fsync(f.fileno())
    return True


def write_owner_record(path: Path, record: OwnerJobRecord) -> None:
    """Replace the record in one step, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(record.model_dump_json())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def move_legacy_owner_record(legacy: Path, path: Path) -> None:
    """Rename a record written under its old name, unless the new one exists.

    A hard link never replaces an existing file, so a record that another
    process just created is kept. A process that loses the race finds the old
    file gone, or the new one there.
    """
    if not legacy.exists():
        return
    try:
        os.link(legacy, path)
    except (FileExistsError, FileNotFoundError):
        pass
    legacy.unlink(missing_ok=True)


def lock_for_submitter(folder: Path, ds_email: str) -> None:
    """Make folder read-only for its submitter.

    The permission file is the nearest one for every file below it, so it
    replaces the submitter's write on the folder above.
    """
    ruleset = RuleSet(rules=[Rule(pattern="**", access=Access(read=[ds_email]))])
    ruleset.save(folder / PERMISSION_FILE_NAME)
