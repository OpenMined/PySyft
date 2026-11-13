"""A job's read permissions: {file in the job's review folder: [emails]}.

The submitter requests them at submission. Only the data owner approves them,
and only within the request.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

from syft_permissions.engine.compiled_rule import USER_PLACEHOLDER

ReadPermissions = dict[str, list[str]]

# Where the submitter puts its request, in the headers of config.yaml.
REQUESTED_READ_PERMISSIONS_HEADER = "requested_read_permissions"

RETURN_CODE_FILENAME = "returncode.txt"
# The folder the job writes its results into.
OUTPUTS = "outputs"
# The files the job's code prints into.
_PRINTED_FILES = ("stdout.txt", "stderr.txt")
# What the runner shares when it is told to share the logs.
LOG_FILES = (*_PRINTED_FILES, RETURN_CODE_FILENAME)
# The files a run writes into the job's review folder.
RUN_FILES = (*LOG_FILES, OUTPUTS)


def job_read_permissions_for(
    readers: list[str], logs: bool, outputs: bool
) -> ReadPermissions:
    """Read for ``readers`` on stdout and stderr if ``logs``, on the outputs if ``outputs``."""
    permissions = {}
    if logs:
        permissions["stdout.txt"] = list(readers)
        permissions["stderr.txt"] = list(readers)
    if outputs:
        permissions[OUTPUTS] = list(readers)
    return permissions


def merge_read_permissions(*permissions: ReadPermissions) -> ReadPermissions:
    """One dict with, per file, every email that any of ``permissions`` names.

    A file that no one names is left out.
    """
    merged: dict[str, set[str]] = {}
    for permission in permissions:
        for file, emails in permission.items():
            merged.setdefault(file, set()).update(emails)
    return {file: sorted(emails) for file, emails in merged.items() if emails}


def validate_read_permissions(value: object) -> ReadPermissions:
    """Return ``value`` as read permissions, with each list sorted.

    Raises:
        TypeError: If ``value`` is not a dict of file names to lists of emails.
        ValueError: If a file is not a plain path inside the job's folder, or
            has no reader.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Read permissions are a dict of file names to lists of emails, "
            f'e.g. {{"stdout.txt": ["ds@example.org"]}}, not {type(value).__name__}.'
        )
    return {file: _check_entry(file, emails) for file, emails in value.items()}


def normalize_file_names(permissions: object) -> object:
    """``permissions`` with each file as a plain path, e.g. "./stderr.txt" -> "stderr.txt".

    For what a submitter types. The strict check after it still refuses ".",
    "..", absolute paths and anything that is not {file: [emails]}.

    Raises:
        ValueError: If two files name the same path.
    """
    if not isinstance(permissions, Mapping):
        return permissions
    normalized: dict = {}
    typed_as: dict = {}
    for file, emails in permissions.items():
        plain = _plain_file_name(file)
        if plain in normalized:
            raise ValueError(f"{typed_as[plain]!r} and {file!r} name the same file.")
        normalized[plain] = emails
        typed_as[plain] = file
    return normalized


def _plain_file_name(file: object) -> object:
    """``file`` stripped, with "/" separators and no "./", "//" or trailing "/"."""
    if not isinstance(file, str):
        return file
    stripped = file.strip()
    # Left empty, so the strict check refuses it.
    if not stripped:
        return stripped
    return str(PurePosixPath(stripped.replace("\\", "/")))


def _check_entry(file: object, emails: object) -> list[str]:
    """``emails`` as the sorted readers of ``file``, after checking both."""
    _check_file(file)
    if not isinstance(emails, (list, tuple)) or not all(
        isinstance(e, str) and e for e in emails
    ):
        raise TypeError(f"The readers of {file!r} must be a list of emails.")
    if not emails:
        raise ValueError(f"{file!r} has no reader.")
    check_no_wildcard_readers(emails)
    return sorted(set(emails))


def check_no_wildcard_readers(emails: Iterable[str]) -> None:
    """Refuse a reader that syft-permissions reads as more than one user.

    Raises:
        ValueError: If a reader is "*", a pattern such as "*@x.org", or "USER".
    """
    wildcards = [e for e in emails if _is_wildcard(e)]
    if wildcards:
        raise ValueError(f"Name each reader by email, not {wildcards}.")


def _check_file(file: object) -> None:
    """Refuse a file that is not one plain path inside the job's folder.

    "." would name the folder itself, and "./stderr.txt" or "stderr.txt/" a
    second name for a file, which the checks on file names would miss.
    """
    if not isinstance(file, str) or not file:
        raise TypeError(f"A read permission names a file as a string, not {file!r}.")
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{file!r} is outside the job's folder.")
    if file == "." or str(path) != file:
        raise ValueError(f"Name {file!r} as a plain path, e.g. 'outputs/result.txt'.")


def _is_wildcard(email: str) -> bool:
    """Matches the patterns of syft_permissions.engine.compiled_rule._user_in_list."""
    return email == USER_PLACEHOLDER or "*" in email or "?" in email


def validate_read_permissions_as_do(value: object) -> ReadPermissions:
    """The valid entries of ``value``, each a plain path in the job's folder.

    For data the submitter wrote: a key is joined to the job's review folder.
    """
    if not isinstance(value, Mapping):
        return {}
    valid = {}
    for file, emails in value.items():
        try:
            valid[file] = _check_entry(file, emails)
        except (TypeError, ValueError):
            continue
    return valid


def restrict_to_request(
    approved: ReadPermissions, requested: ReadPermissions
) -> ReadPermissions:
    """Per file, the emails in ``approved`` that ``requested`` also names."""
    restricted = {}
    for file, emails in approved.items():
        allowed = [email for email in emails if email in requested.get(file, [])]
        if allowed:
            restricted[file] = allowed
    return restricted


def check_approval_reason(reason: object) -> None:
    """Refuse a reason that is not a string, such as a dict of read permissions.

    ``approve({...})`` would record the dict as the reason and approve nothing.
    """
    if reason is not None and not isinstance(reason, str):
        raise TypeError(
            f"reason must be a string, not {type(reason).__name__}. "
            f"Give the read permissions as share_files={{...}}."
        )


LOGS_WARNING = (
    "This approval lets someone read stdout or stderr, as the job wrote them. "
    "The job code can print private data into them. Approve them only for code "
    "that you reviewed for this."
)


def warn_on_logs_release(approved: ReadPermissions, stacklevel: int = 3) -> None:
    """Warn when an approval lets anyone read stdout or stderr."""
    if any(_holds_a_log(file) for file in approved):
        warnings.warn(LOGS_WARNING, UserWarning, stacklevel=stacklevel)


def _holds_a_log(file: str) -> bool:
    """Whether ``file`` is stdout or stderr, or a folder that holds one."""
    return any(PurePosixPath(log).is_relative_to(file) for log in _PRINTED_FILES)


def format_read_permissions(permissions: ReadPermissions) -> str:
    """The permissions for display, e.g. "stdout.txt (a@x.org, b@x.org)", or "none"."""
    parts = [
        f"{file} ({', '.join(emails)})" for file, emails in sorted(permissions.items())
    ]
    return ", ".join(parts) or "none"
