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


@dataclass(frozen=True)
class RequestRef:
    """One api call on disk, in requests/: who it is for, and who sent it.

    A request becomes a job of the same name in inbox/. That job's folders in
    review/ and staging/ also hold the request's state while it waits.
    """

    datasite_email: str  # DO whose datasite holds the request
    ds_email: str  # caller
    job_name: str  # the name of the job the request becomes
    protocol_version: str  # "1"+; requests have no protocol 0 layout

    @property
    def job_ref(self) -> JobRef:
        """The job the request becomes."""
        return JobRef(
            self.datasite_email, self.ds_email, self.job_name, self.protocol_version
        )
