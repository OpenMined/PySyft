from dataclasses import dataclass


class JobStateNotFoundError(FileNotFoundError):
    """Raised when a job's state.yaml does not exist yet."""


@dataclass(frozen=True)
class JobRef:
    """One job on disk: who owns it, who submitted it, and its protocol layout."""

    datasite_email: str  # DO whose datasite holds the job
    ds_email: str  # submitter
    job_name: str
    protocol_version: str  # "0" (no path segment) or "1"+ (v<n> segment)
