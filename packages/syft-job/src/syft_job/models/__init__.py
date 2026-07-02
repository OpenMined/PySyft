from .current_job_state import JobState
from .current_job_submission_metadata import JobSubmissionMetadata
from .job_state_v1 import JobStateV1, JobStatus
from .job_submission_metadata_v1 import JobSubmissionMetadataV1
from .migration import job_registry
from . import schema  # noqa: F401  (registers the current protocol schema on import)

__all__ = [
    # Current-version aliases
    "JobSubmissionMetadata",
    "JobState",
    "JobStatus",
    # Versioned objects
    "JobSubmissionMetadataV1",
    "JobStateV1",
    # Migration registry
    "job_registry",
]
