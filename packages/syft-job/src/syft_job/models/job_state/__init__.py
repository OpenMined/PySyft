from .v0 import JobStateV0, JobStatus

# The current version of the job state object.
JobState = JobStateV0

__all__ = [
    "JobState",
    "JobStateV0",
    "JobStatus",
]
