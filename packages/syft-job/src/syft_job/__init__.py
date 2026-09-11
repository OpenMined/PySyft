# __version__ comes from the installed distribution metadata (see version.py).
from .version import __version__
from .logging_config import configure_package_logger

from .client import BaseJobClient, JobClient, get_client
from .config import SyftJobConfig
from .job import JobInfo, JobsList
from .job_runner import SyftJobRunner, create_runner
from .migrations import job_registry
from .migrations.history import register_historic_schemas
from .models import JobState, JobStatus, JobSubmissionMetadata

# Historic schemas list object versions that must already be registered, which
# happens when the models above are imported.
register_historic_schemas()

__all__ = [
    "__version__",
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

# Last, so the imports stay at the top. Nothing here logs at import time.
configure_package_logger(__name__)
