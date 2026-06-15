__version__ = "0.2.2"

from syft_bg.api import (
    AuthResult,
    AutoApproveResult,
    InitResult,
    InstallationResult,
    StatusResult,
    auto_approve,
    auto_approve_job,
    ensure_running,
    init,
    install,
    list_auto_approvals,
    logs,
    remove_auto_approve,
    restart,
    reset,
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
    "auto_approve",
    "auto_approve_job",
    "AutoApproveResult",
    "list_auto_approvals",
    "remove_auto_approve",
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
