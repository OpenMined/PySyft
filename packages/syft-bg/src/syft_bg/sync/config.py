"""Configuration for the sync service."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from syft_bg.common.config import get_default_paths


class SyncConfig(BaseModel):
    interval: int = 10
    max_retries: int = 3
    retry_backoff: float = 2.0
    do_email: Optional[str] = None
    syftbox_root: Optional[Path] = None
    drive_token_path: Path = Field(
        default_factory=lambda: get_default_paths().drive_token
    )
    sync_state_path: Path = Field(
        default_factory=lambda: get_default_paths().sync_state
    )
    skip_peer_on_patch_version_diff: Optional[bool] = (
        None  # None: value is determined by the role
    )
    force_ignore_peer_version: bool = False
