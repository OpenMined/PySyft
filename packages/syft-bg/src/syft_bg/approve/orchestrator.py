"""Approval orchestrator for auto-approving jobs and peers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from syft_bg.approve.config import AutoApproveConfig
from syft_bg.approve.monitors.job import JobMonitor
from syft_bg.approve.monitors.peer import PeerMonitor
from syft_bg.common.orchestrator import BaseOrchestrator
from syft_bg.common.state import JsonStateManager

if TYPE_CHECKING:
    from syft_client.sync.syftbox_manager import SyftboxManager


class ApprovalOrchestrator(BaseOrchestrator):
    """Orchestrator for job and peer auto-approval service."""

    def __init__(
        self,
        client: SyftboxManager,
        config: AutoApproveConfig,
        config_path: Optional[Path] = None,
    ):
        super().__init__()
        self.client = client
        self.config = config
        self.interval = config.interval
        self._config_path = config_path

        self._state = JsonStateManager(config.approve_state_path)
        self._monitors_initialized = False

    def setup(self) -> None:
        """Verify client and config by initializing monitors."""
        self._init_monitors()

    @classmethod
    def from_client(
        cls,
        client: SyftboxManager,
        interval: int = 5,
    ) -> ApprovalOrchestrator:
        """Create orchestrator from a SyftboxManager client."""
        if not client.has_do_role:
            raise ValueError(
                "ApprovalOrchestrator should only run on Data Owner (DO) side."
            )

        config = AutoApproveConfig.load()
        config.do_email = client.email
        config.syftbox_root = client.syftbox_folder
        config.interval = interval

        return cls(client=client, config=config)

    @classmethod
    def from_config(
        cls,
        config: AutoApproveConfig,
    ) -> ApprovalOrchestrator:
        """Create orchestrator from an AutoApproveConfig."""
        if not config.do_email:
            raise ValueError("Config missing 'do_email' field")
        if not config.syftbox_root:
            raise ValueError("Config missing 'syftbox_root' field")

        # Wait for sync to seed the cache before building SyftboxManager —
        # SyftboxManager.from_config triggers _load_file_hashes_from_disk,
        # which races sync's identical replay if both run cold-start.
        cls._wait_for_sync_ready(label="Approve")

        from syft_client.sync.environments.environment import Environment
        from syft_client.sync.syftbox_manager import SyftboxManager
        from syft_client.sync.utils.syftbox_utils import check_env

        env = check_env()
        if env == Environment.COLAB:
            client = SyftboxManager.for_colab(
                email=config.do_email,
                has_do_role=True,
                skip_peer_on_patch_version_diff=config.skip_peer_on_patch_version_diff,
                force_ignore_peer_version=config.force_ignore_peer_version,
            )
        else:
            client = SyftboxManager.for_jupyter(
                email=config.do_email,
                has_do_role=True,
                token_path=config.drive_token_path,
                skip_peer_on_patch_version_diff=config.skip_peer_on_patch_version_diff,
                force_ignore_peer_version=config.force_ignore_peer_version,
            )

        return cls(client=client, config=config)

    def _collect_auto_approve_emails(self) -> set[str]:
        """Collect peer emails from all auto-approval objects."""
        emails: set[str] = set()
        for obj in self.config.auto_approvals.objects.values():
            emails.update(obj.peers)
        return emails

    def _init_monitors(self):
        """Initialize job and peer monitors."""
        if self._monitors_initialized:
            return

        if self.config.auto_approvals.enabled:
            self._job_monitor = JobMonitor(
                client=self.client,
                config_path=self._config_path,
                state=self._state,
                verbose=True,
            )

        self.config.peers.auto_approve_emails = list(
            self._collect_auto_approve_emails()
        )
        if self.config.peers.auto_approve_emails or self.config.peers.approved_domains:
            self._peer_monitor = PeerMonitor(
                client=self.client,
                config=self.config.peers,
                state=self._state,
                verbose=True,
            )

        self._monitors_initialized = True

    def _print_startup_info(self):
        """Print startup info for approval service."""
        print("Starting approval daemon...")
        print(f"  DO: {self.config.do_email}")
        print(f"  SyftBox: {self.config.syftbox_root}")
        print(f"  Interval: {self.config.interval}s")
        print(
            f"  Auto-approvals: {'enabled' if self.config.auto_approvals.enabled else 'disabled'}"
        )
        emails = self.config.peers.auto_approve_emails
        domains = self.config.peers.approved_domains
        print(f"  Peer approval: {len(emails)} emails, {len(domains)} domains")
        print()
