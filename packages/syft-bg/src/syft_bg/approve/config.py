"""Configuration for the approval service."""

from pathlib import Path
from typing import Optional

import yaml
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

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AutoApproveConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = get_default_paths().config
        else:
            config_path = Path(config_path).expanduser()

        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        common = {k: v for k, v in data.items() if not isinstance(v, dict)}
        approve_section = data.get("approve", {})

        auto_approvals_data = approve_section.get("auto_approvals", {})
        peers_data = approve_section.get("peers", {})

        kwargs: dict = {
            "interval": approve_section.get("interval", 5),
            "auto_approvals": AutoApprovalsConfig(**auto_approvals_data)
            if auto_approvals_data
            else AutoApprovalsConfig(),
            "peers": PeerApprovalConfig(**peers_data)
            if peers_data
            else PeerApprovalConfig(),
            "skip_peer_on_patch_version_diff": approve_section.get(
                "skip_peer_on_patch_version_diff"
            ),
            "force_ignore_peer_version": approve_section.get(
                "force_ignore_peer_version", False
            ),
        }
        if common.get("do_email"):
            kwargs["do_email"] = common["do_email"]
        if common.get("syftbox_root"):
            kwargs["syftbox_root"] = Path(common["syftbox_root"]).expanduser()
        if common.get("drive_token_path"):
            kwargs["drive_token_path"] = Path(common["drive_token_path"]).expanduser()

        return cls(**kwargs)

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = get_default_paths().config
        else:
            config_path = Path(config_path).expanduser()

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

        if self.do_email:
            data["do_email"] = self.do_email
        if self.syftbox_root:
            data["syftbox_root"] = str(self.syftbox_root)
        if self.drive_token_path:
            data["drive_token_path"] = str(self.drive_token_path)

        data["approve"] = {
            "interval": self.interval,
            "auto_approvals": self.auto_approvals.model_dump(),
            "peers": self.peers.model_dump(),
            "skip_peer_on_patch_version_diff": self.skip_peer_on_patch_version_diff,
            "force_ignore_peer_version": self.force_ignore_peer_version,
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


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
