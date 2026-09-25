"""Configuration for the approval service."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from syft_rds.apis.models import ApiDefinition, FileEntry

from syft_bg.common.config import get_default_paths


# The api format is shared with the client, which reads it on the DS side.
AutoApprovalObj = ApiDefinition


class AutoApprovalsConfig(BaseModel):
    """Configuration for auto-approval objects.

    The objects themselves live in SyftBox, see syft_bg.approve.api_store.
    """

    enabled: bool = True


class PeerApprovalConfig(BaseModel):
    """Configuration for peer auto-approval."""

    enabled: bool = False
    approved_domains: list[str] = Field(default_factory=list)
    auto_share_datasets: list[str] = Field(default_factory=list)
    auto_approve_emails: list[str] = Field(default_factory=list, exclude=True)


class AutoApproveConfig(BaseModel):
    """Main configuration for the approval service."""

    do_email: Optional[str] = None
    syftbox_root: Optional[Path] = None
    drive_token_path: Path = Field(
        default_factory=lambda: get_default_paths().drive_token
    )
    gmail_token_path: Path = Field(
        default_factory=lambda: get_default_paths().gmail_token
    )
    approve_state_path: Path = Field(
        default_factory=lambda: get_default_paths().approve_state
    )
    notify_state_path: Path = Field(
        default_factory=lambda: get_default_paths().notify_state
    )
    interval: int = 5
    auto_approvals: AutoApprovalsConfig = Field(default_factory=AutoApprovalsConfig)
    peers: PeerApprovalConfig = Field(default_factory=PeerApprovalConfig)
    skip_peer_on_patch_version_diff: Optional[bool] = (
        None  # None: value is determined by the role
    )
    force_ignore_peer_version: bool = False


# --- Backwards-compatible aliases (deprecated, will be removed) ---

ScriptEntry = FileEntry
ScriptRule = FileEntry


class PeerApprovalEntry(BaseModel):
    """Deprecated: use AutoApprovalObj instead."""

    mode: str = "strict"
    scripts: list[FileEntry] = Field(default_factory=list)


PeerJobConfig = PeerApprovalEntry


class JobApprovalConfig(BaseModel):
    """Deprecated: use AutoApprovalsConfig instead."""

    enabled: bool = True
    peers: dict[str, PeerApprovalEntry] = Field(default_factory=dict)
