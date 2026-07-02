__version__ = "0.1.25"

from .client import BaseJobClient, JobClient, get_client
from .config import SyftJobConfig
from .job import JobInfo, JobsList
from .job_runner import SyftJobRunner, create_runner
from .migrations import job_registry
from .migrations import schema  # noqa: F401  (registers the current protocol schema)
from .models import JobState, JobStatus, JobSubmissionMetadata

__all__ = [
    # SyftBox job system
    "BaseJobClient",
    "JobClient",
    "get_client",
    "SyftJobConfig",
    "SyftJobRunner",
    "create_runner",
    # Job types
    "JobInfo",
    "JobsList",
    # Models
    "JobSubmissionMetadata",
    "JobState",
    "JobStatus",
    # Migration registry
    "job_registry",
]
