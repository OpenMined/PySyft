"""A job's read permissions: {file in the job's review folder: [emails]}.

The submitter requests them at submission. Only the data owner approves them,
and only within the request.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

from .traceback_capture import FRAMES_FILENAME

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
# The run files a read permission can name.
READABLE_RUN_FILES = (*_PRINTED_FILES, FRAMES_FILENAME, RETURN_CODE_FILENAME, OUTPUTS)


def read_permissions_for(
    readers: list[str], logs: bool, outputs: bool
) -> ReadPermissions:
    """Read on stdout and stderr, and on the outputs, for ``readers``."""
    files = (*(_PRINTED_FILES if logs else ()), *((OUTPUTS,) if outputs else ()))
    return {file: sorted(set(readers)) for file in files} if readers else {}


def merge_read_permissions(*permissions: ReadPermissions) -> ReadPermissions:
    """One dict with, per file, every email that any of ``permissions`` names."""
    merged: dict[str, set[str]] = {}
    for permission in permissions:
        for file, emails in permission.items():
            merged.setdefault(file, set()).update(emails)
    return {file: sorted(emails) for file, emails in merged.items()}


def check_one_way(dict_arg: str, given: object, flags: dict[str, bool]) -> None:
    """Refuse ``dict_arg`` together with any of the ``flags`` that is set.

    Raises:
        ValueError: If the dict and a flag are both given.
    """
    set_flags = [name for name, value in flags.items() if value]
    if given is not None and set_flags:
        raise ValueError(f"Give {dict_arg} or {', '.join(set_flags)}, not both.")


def check_read_permissions(value: object) -> ReadPermissions:
    """Return ``value`` as read permissions, with each list sorted.

    Raises:
        TypeError: If ``value`` is not a dict of file names to lists of emails.
        ValueError: If a file is not in READABLE_RUN_FILES, or has no reader.
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
    if file not in READABLE_RUN_FILES:
        raise ValueError(
            f"{file!r} is not a file a read permission can name. "
            f"Name one of: {', '.join(READABLE_RUN_FILES)}."
        )
    if not isinstance(emails, (list, tuple)) or not all(
        isinstance(e, str) and e for e in emails
    ):
        raise TypeError(f"The readers of {file!r} must be a list of emails.")
    if not emails:
        raise ValueError(f"{file!r} has no reader.")
    return sorted(set(emails))


def valid_read_permissions(value: object) -> ReadPermissions:
    """The valid entries of ``value``, without the rest. For data the submitter wrote."""
    if not isinstance(value, Mapping):
        return {}
    valid = {}
    for file, emails in value.items():
        try:
            valid[file] = _check_readers(file, emails)
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
            f"Give the read permissions as read_permissions={{...}}."
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
