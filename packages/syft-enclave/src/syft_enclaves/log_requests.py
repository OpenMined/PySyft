"""Data owners asking the enclave to share a job's logs after it ran.

Each data owner writes its own request file in the job's review folder. The
enclave shares a log only when every data owner made the same request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field, field_validator
from syft_job.read_permissions import check_no_wildcard_readers

# The logs a data owner can request.
REQUESTABLE_LOGS = ("stdout.txt", "stderr.txt")


class LogRequest(BaseModel):
    """Who should read one log, and how many of its first chars."""

    # None means the data owners the enclave is configured with. The DS names
    # the job's datasets, so a default from those could include the DS.
    readers: Optional[list[str]] = None
    # None shares the whole log.
    n_chars: Optional[int] = Field(default=None, gt=0)

    @field_validator("readers", mode="before")
    @classmethod
    def _one_reader_as_list(cls, readers: object) -> object:
        return [readers] if isinstance(readers, str) else readers

    @field_validator("readers")
    @classmethod
    def _sorted_unique(cls, readers: Optional[list[str]]) -> Optional[list[str]]:
        if readers is None:
            return None
        if not readers:
            raise ValueError("A log request needs at least one reader.")
        check_no_wildcard_readers(readers)
        return sorted(set(readers))


class RequestedPermissionsUpdate(BaseModel):
    """One data owner's log requests for one job, keyed by log file."""

    party: str
    logs: dict[str, LogRequest] = Field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json")))

    @classmethod
    def load_json(cls, path: Path) -> RequestedPermissionsUpdate:
        return cls(**json.loads(path.read_text()))


def log_request_file_name(do_email: str) -> str:
    return f"{do_email}_requested_permissions_update.json"


def load_log_request(
    review_dir: Path, party: str
) -> Optional[RequestedPermissionsUpdate]:
    """The request file of ``party``, or None when missing, unreadable or another party's."""
    try:
        request = RequestedPermissionsUpdate.load_json(
            review_dir / log_request_file_name(party)
        )
    except (OSError, ValueError):
        return None
    if request.party.casefold() != party.casefold():
        return None
    return request


def logs_requested_by_all_parties(
    review_dir: Path, parties: Iterable[str]
) -> dict[str, LogRequest]:
    """Per log, the request that every party made the same.

    Empty when a party has no valid request file, or there are no parties.
    """
    requests = [load_log_request(review_dir, p) for p in parties]
    if not requests or any(r is None for r in requests):
        return {}
    first, *others = requests
    return {
        log_file: request
        for log_file, request in first.logs.items()
        if log_file in REQUESTABLE_LOGS
        and all(other.logs.get(log_file) == request for other in others)
    }


def log_head_name(log_file: str, n_chars: int) -> str:
    """The copy of the first ``n_chars`` chars of a log, e.g. stderr_100.txt."""
    path = Path(log_file)
    return f"{path.stem}_{n_chars}{path.suffix}"


def write_log_head(
    review_dir: Path, log_file: str, n_chars: Optional[int]
) -> Optional[str]:
    """The file to share for a request: the log, or a copy of its first ``n_chars`` chars.

    None when the job wrote no such log.
    """
    log = review_dir / log_file
    if not log.exists():
        return None
    if n_chars is None:
        return log_file
    head = review_dir / log_head_name(log_file, n_chars)
    if not head.exists():
        head.write_text(log.read_text(errors="replace")[:n_chars])
    return head.name


def longest_log_head(review_dir: Path, log_file: str) -> Optional[Path]:
    """The longest copy of the first chars of ``log_file`` in ``review_dir``, or None."""
    stem, suffix = Path(log_file).stem, Path(log_file).suffix
    heads = {}
    for path in review_dir.glob(f"{stem}_*{suffix}"):
        n_chars = path.stem.removeprefix(f"{stem}_")
        if n_chars.isdigit():
            heads[int(n_chars)] = path
    return heads[max(heads)] if heads else None
