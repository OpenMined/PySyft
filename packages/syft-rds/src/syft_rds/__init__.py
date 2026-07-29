"""syft-rds: Remote Data Science product composed on top of syft-client."""

from syft_rds.client import SyftRDSClient
from syft_rds.config import SyftRDSClientConfig
from syft_rds.job_auto_approval import auto_approve_and_run_jobs, job_matches_criteria
from syft_rds.login import login_do, login_ds

# Generic sync-stack helpers re-exported so consumers that compose the RDS product
# (e.g. syft-bg) have a single integration surface and need not import from
# syft_client.sync internals directly.
from syft_client.sync.syftbox_manager import get_jupyter_default_syftbox_folder
from syft_client.sync.utils.path_filters import is_normal_syncable_path
from syft_client.sync.environments.environment import Environment
from syft_client.sync.utils.syftbox_utils import check_env

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
