"""Configuration utilities shared across services."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

COLAB_DRIVE_PATH = Path("/content/drive/MyDrive")
CREDS_DIR_NAME = "syft-bg"


def get_syftbg_dir() -> Path:
    """Get the credentials directory, handling Colab vs local environments."""
    if COLAB_DRIVE_PATH.exists():
        return COLAB_DRIVE_PATH / CREDS_DIR_NAME
    return Path.home() / f".{CREDS_DIR_NAME}"


@dataclass
class DefaultPaths:
    """Default paths for all services."""

    # Shared paths
    config: Path
    credentials: Path
    gmail_token: Path
    drive_token: Path

    # Notify service paths
    notify_state: Path
    notify_pid: Path
    notify_log: Path

    # Approve service paths
    approve_state: Path
    approve_pid: Path
    approve_log: Path
    auto_approvals_dir: Path

    # Email approve service paths
    email_approve_state: Path
    email_approve_pid: Path
    email_approve_log: Path

    # Sync service paths
    sync_state: Path
    sync_pid: Path
    sync_log: Path

    # Setup state paths (per-service)
    notify_setup_state: Path
    approve_setup_state: Path
    email_approve_setup_state: Path
    sync_setup_state: Path


def get_default_paths() -> DefaultPaths:
    """Get default paths for all services."""
    creds = get_syftbg_dir()
    return DefaultPaths(
        # Shared
        config=creds / "config.yaml",
        credentials=creds / "credentials.json",
        gmail_token=creds / "token.json",
        drive_token=creds / "token.json",
        # Notify
        notify_state=creds / "notify" / "state.json",
        notify_pid=creds / "notify" / "daemon.pid",
        notify_log=creds / "notify" / "daemon.log",
        # Approve
        approve_state=creds / "approve" / "state.json",
        approve_pid=creds / "approve" / "daemon.pid",
        approve_log=creds / "approve" / "daemon.log",
        auto_approvals_dir=creds / "auto_approvals",
        # Email approve
        email_approve_state=creds / "email_approve" / "state.json",
        email_approve_pid=creds / "email_approve" / "daemon.pid",
        email_approve_log=creds / "email_approve" / "daemon.log",
        # Sync
        sync_state=creds / "sync" / "state.json",
        sync_pid=creds / "sync" / "daemon.pid",
        sync_log=creds / "sync" / "daemon.log",
        # Setup state
        notify_setup_state=creds / "notify" / "setup_state.json",
        approve_setup_state=creds / "approve" / "setup_state.json",
        email_approve_setup_state=creds / "email_approve" / "setup_state.json",
        sync_setup_state=creds / "sync" / "setup_state.json",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save data to YAML config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
