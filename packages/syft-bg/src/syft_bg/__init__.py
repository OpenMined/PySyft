__version__ = "0.2.2"

from syft_job.logging_config import configure_package_logger

from syft_bg.api import (
    AuthResult,
    ApiResult,
    InitResult,
    InstallationResult,
    StatusResult,
    create_api,
    create_api_from_job,
    ensure_running,
    init,
    install,
    list_apis,
    logs,
    remove_api,
    restart,
    reset,
    run_policy,
    set_policies,
    start,
    stop,
    uninstall,
)
from syft_bg.services import ServiceManager

__all__ = [
    "ServiceManager",
    "InitResult",
    "InstallationResult",
    "AuthResult",
    "create_api",
    "create_api_from_job",
    "ApiResult",
    "list_apis",
    "remove_api",
    "set_policies",
    "run_policy",
    "status",
    "StatusResult",
    "start",
    "stop",
    "restart",
    "logs",
    "ensure_running",
    "init",
    "install",
    "reset",
    "uninstall",
]


def __getattr__(name: str):
    if name == "status":
        from syft_bg.api import status

        return status()

    if name == "config":
        from syft_bg.common.syft_bg_config import SyftBgConfig

        try:
            return SyftBgConfig.from_path()
        except FileNotFoundError:
            print("No config file found, run syft_bg.init() first")
            return
    raise AttributeError(f"module 'syft_bg' has no attribute {name!r}")


# Last, so the imports stay at the top. Nothing here logs at import time.
configure_package_logger(__name__)
