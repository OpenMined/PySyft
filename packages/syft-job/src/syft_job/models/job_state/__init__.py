from .v1 import JobStateV1, JobStatus

# The current version of the job state object.
JobState = JobStateV1

__all__ = [
    "JobState",
    "JobStateV1",
    "JobStatus",
]
