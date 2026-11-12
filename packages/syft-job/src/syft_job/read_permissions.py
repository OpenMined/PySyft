"""A job's read permissions: {file in the job's review folder: [emails]}.

The submitter requests them at submission. Only the data owner approves them,
and only within the request.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import PurePosixPath

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
        ValueError: If a file has no reader.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Read permissions are a dict of file names to lists of emails, "
            f'e.g. {{"stdout.txt": ["ds@example.org"]}}, not {type(value).__name__}.'
        )
    return {file: _check_readers(file, emails) for file, emails in value.items()}


def _check_readers(file: object, emails: object) -> list[str]:
    if not isinstance(file, str) or not file:
        raise TypeError(f"A read permission names a file as a string, not {file!r}.")
    if not isinstance(emails, (list, tuple)) or not all(
        isinstance(e, str) and e for e in emails
    ):
        raise TypeError(f"The readers of {file!r} must be a list of emails.")
    if not emails:
        raise ValueError(f"{file!r} has no reader.")
    return sorted(set(emails))


def validate_read_permissions_as_do(value: object) -> ReadPermissions:
    """The valid entries of ``value`` that stay in the job's folder.

    For data the submitter wrote: a key is joined to the job's review folder.
    """
    if not isinstance(value, Mapping):
        return {}
    valid = {}
    for file, emails in value.items():
        if not _inside_job_folder(file):
            continue
        try:
            valid[file] = _check_readers(file, emails)
        except (TypeError, ValueError):
            continue
    return valid


def _inside_job_folder(file: object) -> bool:
    """Whether ``file`` is a relative path that stays in the job's folder."""
    if not isinstance(file, str):
        return False
    path = PurePosixPath(file)
    return not path.is_absolute() and ".." not in path.parts


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
    if any(file in approved for file in _PRINTED_FILES):
        warnings.warn(LOGS_WARNING, UserWarning, stacklevel=stacklevel)


def format_read_permissions(permissions: ReadPermissions) -> str:
    """The permissions for display, e.g. "stdout.txt (a@x.org, b@x.org)", or "none"."""
    parts = [
        f"{file} ({', '.join(emails)})" for file, emails in sorted(permissions.items())
    ]
    return ", ".join(parts) or "none"
