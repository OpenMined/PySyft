"""What a DS writes to call an api: request.yaml in requests/<ds>/v1/<name>/."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import JobSubmissionMetadata
from .submission import API_HEADER, REQUEST_FILE_NAME

# Files the DO writes into the job a request becomes, so a call can't send them.
RESERVED_CALL_FILES = frozenset({REQUEST_FILE_NAME, "config.yaml"})


class RequestMetadata(BaseModel):
    """A call of an api, as the caller describes it.

    Who sent it, to whom, and under which name come from its path (RequestRef),
    never from this file. When the DO turns the request into a job, it builds
    the job's config.yaml from this with to_job_metadata().
    """

    api: str
    submitted_at: datetime
    # The files the call sends, relative to the job's root.
    files: list[str] = Field(default_factory=list)
    # The caller's hash of those files, so the DO knows when all arrived.
    files_hash: str

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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> RequestMetadata:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})
