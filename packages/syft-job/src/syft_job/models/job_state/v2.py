from __future__ import annotations

from pydantic import Field

from ...migrations import job_registry
from .v1 import JobStateV1


class JobStateV2(JobStateV1):
    """V2 adds the job's read permissions, as {file in the job's review folder: [emails]}.

    The data owner copies the request from the submission at receipt, and sets the
    approved permissions when it approves. The submitter never writes this file.
    """

    version: str = "2"

    requested_read_permissions: dict[str, list[str]] = Field(default_factory=dict)
    approved_read_permissions: dict[str, list[str]] = Field(default_factory=dict)


@job_registry.migration("JobState", "1", "2")
def _job_state_v1_to_v2(obj: JobStateV1) -> JobStateV2:
    return JobStateV2.model_validate(
        obj.model_dump(exclude={"canonical_name", "version"})
    )


@job_registry.migration("JobState", "2", "1")
def _job_state_v2_to_v1(obj: JobStateV2) -> JobStateV1:
    dropped = {
        "canonical_name",
        "version",
        "requested_read_permissions",
        "approved_read_permissions",
    }
    return JobStateV1.model_validate(obj.model_dump(exclude=dropped))
