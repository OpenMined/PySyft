from .v1 import JobStateV1, JobStatus
from .v2 import JobStateV2

# The current version of the job state object.
JobState = JobStateV2

__all__ = [
    "JobState",
    "JobStateV1",
    "JobStateV2",
    "JobStatus",
]
