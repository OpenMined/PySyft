"""Configuration for the approval service."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from syft_bg.common.config import get_default_paths


class FileEntry(BaseModel):
    """A file stored in the auto-approvals directory with its hash."""

    relative_path: str  # e.g. "subdir/main.py"
    path: str  # e.g. "~/.syft-bg/auto_approvals/my_analysis/main.py"
    hash: str  # e.g. "sha256:abc123..."

    @classmethod
    def from_file(cls, relative_path: str, path: str | Path) -> "FileEntry":
        """Create a FileEntry from an existing file, computing its hash."""
        import hashlib

        p = Path(path)
        content = p.read_text(encoding="utf-8")
        file_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        return cls(relative_path=relative_path, path=str(p), hash=file_hash)


class AutoApprovalObj(BaseModel):
    """An auto-approval object bundling content-matched files, name-only files, and peers."""

    file_contents: list[FileEntry] = Field(
        default_factory=list
    )  # files matched by content+hash
    file_paths: list[str] = Field(default_factory=list)  # files matched by path only
    peers: list[str] = Field(default_factory=list)  # peer emails


class AutoApprovalsConfig(BaseModel):
    """Configuration for auto-approval objects."""

    enabled: bool = True
    objects: dict[str, AutoApprovalObj] = Field(default_factory=dict)


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
