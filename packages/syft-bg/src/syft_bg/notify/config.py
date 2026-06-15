"""Configuration for notification service."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from syft_bg.common.config import get_default_paths


class NotifyConfig(BaseModel):
    """Configuration for the notification service."""

    do_email: Optional[str] = None
    syftbox_root: Optional[Path] = None
    drive_token_path: Optional[Path] = Field(
        default_factory=lambda: get_default_paths().drive_token
    )
    gmail_token_path: Optional[Path] = Field(
        default_factory=lambda: get_default_paths().gmail_token
    )
    credentials_path: Optional[Path] = Field(
        default_factory=lambda: get_default_paths().credentials
    )
    notify_state_path: Path = Field(
        default_factory=lambda: get_default_paths().notify_state
    )
    sync_state_path: Path = Field(
        default_factory=lambda: get_default_paths().sync_state
    )
    interval: int = 5
    monitor_jobs: bool = True
    monitor_peers: bool = True
    heartbeat_enabled: bool = True
    heartbeat_interval: int = 86400
