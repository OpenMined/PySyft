from __future__ import annotations

from datetime import datetime

from pydantic import Field
from syft_migration import MigratableObject

from ...migrations import job_registry
from ...submission import API_HEADER
from ..job_submission_metadata import JobSubmissionMetadata


class RequestMetadataV1(MigratableObject, registry=job_registry):
    """A call of an api, stored as request.yaml in requests/<ds>/v<n>/<name>/.

    Who sent it, to whom, and under which name come from its path (RequestRef),
    never from this file.
    """

    canonical_name: str = "RequestMetadata"
    version: str = "1"

    api: str
    submitted_at: datetime
    # The files the call sends, relative to the job's root.
    files: list[str] = Field(default_factory=list)
    # The caller's hash of those files, so the DO knows when all arrived.
    files_hash: str

    def disk_dict(self) -> dict:
        """The on-disk form of the request."""
        return self.model_dump(mode="json")

    def to_job_metadata(
        self, name: str, submitted_by: str, datasite_email: str
    ) -> JobSubmissionMetadata:
        """The config.yaml of the job the request becomes."""
        return JobSubmissionMetadata(
            name=name,
            type="python",
            submitted_by=submitted_by,
            datasite_email=datasite_email,
            submitted_at=self.submitted_at,
            files=self.files,
            headers={API_HEADER: self.api},
        )
