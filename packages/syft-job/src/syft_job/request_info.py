"""An api call that has not become a job: the request a DS wrote to requests/."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .job_ref import JobRef, RequestRef
from .job_storage import JobStateNotFoundError
from .models import JobStatus, RequestMetadata

if TYPE_CHECKING:
    from .client import JobClient
    from .job import JobInfo


class RequestInfo:
    """A call of an api, until the DO turns it into a job or rejects it.

    A rejected request keeps its folder and gets a rejected state. An accepted
    one becomes a job of the same name, found with .job, and its folder is
    removed.
    """

    def __init__(self, ref: RequestRef, metadata: RequestMetadata, client: JobClient):
        self._ref = ref
        self.metadata = metadata
        self._client = client

    @property
    def name(self) -> str:
        """The name of the job the request becomes."""
        return self._ref.job_name

    @property
    def api_name(self) -> str:
        return self.metadata.api

    @property
    def submitted_by(self) -> str:
        return self._ref.ds_email

    @property
    def datasite_owner_email(self) -> str:
        return self._ref.datasite_email

    @property
    def request_path(self) -> Path:
        return self._client.manager.request_dir(self._ref)

    @property
    def status(self) -> str:
        """received, rejected, or accepted once it became a job."""
        if self.job is not None:
            return "accepted"
        try:
            state = self._client.manager.read_state(self._job_ref)
        except JobStateNotFoundError:
            return JobStatus.RECEIVED.value
        return state.status.value

    @property
    def review_reason(self) -> Optional[str]:
        """Why the DO rejected the request, if it did."""
        try:
            return self._client.manager.read_state(self._job_ref).review_reason
        except JobStateNotFoundError:
            return None

    @property
    def job(self) -> Optional[JobInfo]:
        """The job the request became, once the DO accepted it."""
        if not self._client.manager.submission_dir(self._job_ref).exists():
            return None
        return self._client._job_info(self._job_ref)

    @property
    def _job_ref(self) -> JobRef:
        return self._ref.job_ref

    def __repr__(self) -> str:
        reason = f", reason={self.review_reason!r}" if self.status == "rejected" else ""
        return (
            f"RequestInfo(name={self.name!r}, api={self.api_name!r}, "
            f"datasite={self.datasite_owner_email!r}, status={self.status!r}{reason})"
        )
