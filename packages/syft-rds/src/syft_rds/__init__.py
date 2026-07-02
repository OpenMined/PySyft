"""syft-rds: Remote Data Science product composed on top of syft-client."""

from syft_rds.client import SyftRDSClient
from syft_rds.login import login_do, login_ds

__all__ = ["SyftRDSClient", "login_do", "login_ds"]
