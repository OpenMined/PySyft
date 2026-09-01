"""syft-rds: Remote Data Science product composed on top of syft."""

from syft_job.logging_config import configure_package_logger

from syft_rds.client import SyftRDSClient
from syft_rds.config import SyftRDSClientConfig
from syft_rds.job_auto_approval import auto_approve_and_run_jobs, job_matches_criteria
from syft_rds.login import login_do, login_ds

# Generic sync-stack helpers re-exported so consumers that compose the RDS product
# (e.g. syft-bg) have a single integration surface and need not import from
# syft.sync internals directly.
from syft.sync.syftbox_manager import get_jupyter_default_syftbox_folder
from syft.sync.utils.path_filters import is_normal_syncable_path
from syft.sync.environments.environment import Environment
from syft.sync.utils.syftbox_utils import check_env

__all__ = [
    "SyftRDSClient",
    "SyftRDSClientConfig",
    "login_do",
    "login_ds",
    "auto_approve_and_run_jobs",
    "job_matches_criteria",
    "get_jupyter_default_syftbox_folder",
    "is_normal_syncable_path",
    "Environment",
    "check_env",
]

# Last, so the imports stay at the top. Nothing here logs at import time.
configure_package_logger(__name__)
