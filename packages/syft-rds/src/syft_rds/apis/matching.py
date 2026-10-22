"""Whether a job submission matches an api.

The DO's approve service uses this to approve a job, and the job runner uses
it again before the api's policies run, so both read one definition of a match.
"""

import hashlib
import json
from pathlib import Path

from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from syft_rds.apis.api import callable_layout
from syft_rds.apis.args import bind_args
from syft_rds.apis.models import (
    ApiDefinition,
    FileEntry,
    list_api_names,
    load_api_definition,
)

RUN_SCRIPT_PATH = "run.sh"
REQUEST_CONFIG_PATH = "config.yaml"


def compute_file_hash(file_path: Path) -> str | None:
    """Compute SHA256 hash of a file's content."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        return None


def hash_matches(actual_hash: str, expected_hash: str) -> bool:
    """Compare a file digest against a configured one.

    The compare is over the whole digest. A prefix compare let an empty or
    truncated configured hash match any file.
    """
    if expected_hash.startswith("sha256:"):
        expected = expected_hash[7:]
    else:
        expected = expected_hash
    return bool(expected) and actual_hash == expected


def content_matches(job_file: Path, stored_path: str) -> bool:
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


def submission_files(root: Path) -> dict[str, Path]:
    """Every file in a submission as {path relative to its root: abs_path}.

    The root holds `code/`, `run.sh` and `config.yaml`. `run.sh` is the only
    file the runner executes, so it must be in this set. The exact name
    `syft.pub.yaml` is left out, because the permission layer writes it. Any
    other spelling stays in, and is refused as an extra file: an exclusion from
    this inventory is a file nobody reviews, so keep it as narrow as the writer.
    """
    files: dict[str, Path] = {}
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and f.name != PERMISSION_FILE_NAME:
                files[f.relative_to(root).as_posix()] = f
    return files


def match_submission(root: Path, definition: ApiDefinition) -> tuple[bool, str]:
    """Whether the submission at root matches the api. Returns (match, reason).

    Two-step validation for each content-matched file:
    1. Hash must match a file entry in the api
    2. Content must match the stored copy

    All other files must be in the file_paths allowlist.
    """
    expected_contents = {e.relative_path: e for e in definition.file_contents}
    all_expected_paths = set(expected_contents) | set(definition.file_paths)
    job_files = submission_files(root)

    # run.sh is the only file that executes, so a name in the allowlist is not
    # enough for it. An api written before run.sh was validated pins nothing
    # for it and is refused here.
    if RUN_SCRIPT_PATH not in expected_contents:
        return False, (
            f"approval object pins no content for {RUN_SCRIPT_PATH}, the file "
            f"the runner executes. Recreate it from a job you have reviewed "
            f"with syft_bg.auto_approve_job(job)."
        )

    names_reason = _names_mismatch(set(job_files), all_expected_paths)
    if names_reason:
        return False, names_reason

    for rel_path, entry in expected_contents.items():
        reason = _content_mismatch(rel_path, job_files[rel_path], entry)
        if reason:
            return False, reason
    return True, "ok"


def matching_apis(apis_dir: Path, root: Path, submitter: str) -> tuple[list[str], str]:
    """The apis shared with submitter that the submission at root matches.

    Returns their names, and why the last api that didn't match failed. An api
    with no peers is shared with every peer.
    """
    matches: list[str] = []
    last_reason = f"no auto-approval objects match peer: {submitter}"
    for name in list_api_names(apis_dir):
        definition = load_api_definition(apis_dir / name)
        if definition.peers and submitter not in definition.peers:
            continue
        match, reason = match_submission(root, definition)
        if match:
            matches.append(name)
        else:
            last_reason = f"[{name}] {reason}"
    return matches, last_reason


def check_request(
    root: Path, definition: ApiDefinition, submitter: str
) -> tuple[bool, str]:
    """Whether the request at root may call the api. Returns (ok, reason).

    A request holds no code, so nothing is matched by content. It may hold only
    the api's name-only files, its params must fit the api's arguments, and
    the submitter must be one of the api's peers.
    """
    if definition.peers and submitter not in definition.peers:
        return False, f"the api is not shared with {submitter}"
    sent = set(submission_files(root)) - {REQUEST_CONFIG_PATH}
    not_allowed = sent - set(definition.file_paths)
    if not_allowed:
        return False, f"files the api doesn't take: {sorted(not_allowed)}"
    return _check_params(root, definition)


def _check_params(root: Path, definition: ApiDefinition) -> tuple[bool, str]:
    """Whether the request's params json binds to the api's arguments."""
    layout = callable_layout(definition)
    if layout is None:
        return False, "the api can't be called with arguments"
    try:
        params = json.loads((root / layout.params_file).read_text())
        bind_args("api", definition.args, (), params)
    except (OSError, ValueError, TypeError) as e:
        return False, f"params don't fit the api: {e}"
    return True, "ok"


def _names_mismatch(actual: set[str], expected: set[str]) -> str | None:
    """Why the submission's file names differ from the api's, or None."""
    if actual == expected:
        return None
    parts = []
    if expected - actual:
        parts.append(f"missing files: {expected - actual}")
    if actual - expected:
        parts.append(f"extra files: {actual - expected}")
    return f"job files do not match expected filenames: {'; '.join(parts)}"


def _content_mismatch(rel_path: str, job_file: Path, entry: FileEntry) -> str | None:
    """Why a pinned file differs from the api's copy, or None."""
    submitted_hash = compute_file_hash(job_file)
    if submitted_hash is None:
        return f"could not read file: {rel_path}"
    if not hash_matches(submitted_hash, entry.hash):
        return (
            f"file hash mismatch for {rel_path}: expected {entry.hash}, "
            f"got sha256:{submitted_hash}"
        )
    if not content_matches(job_file, entry.path):
        return f"file content mismatch for {rel_path} against stored copy"
    return None
