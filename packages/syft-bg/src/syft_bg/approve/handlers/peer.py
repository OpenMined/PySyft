"""Peer approval handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Protocol

from syft_bg.approve.config import PeerApprovalConfig

if TYPE_CHECKING:
    from syft_client.sync.syftbox_manager import SyftboxManager


class StateManager(Protocol):
    """Protocol for state management."""

    def mark_approved(self, key: str, info: str) -> None: ...
    def was_approved(self, key: str) -> bool: ...


class PeerApprovalHandler:
    """Handles peer approval logic."""

    def __init__(
        self,
        client: SyftboxManager,
        config: PeerApprovalConfig,
        state: Optional[StateManager] = None,
        on_approve: Optional[Callable[[str], None]] = None,
        verbose: bool = True,
    ):
        self.client = client
        self.config = config
        self.state = state
        self.on_approve = on_approve
        self.verbose = verbose

    def _get_pending_peers(self) -> list[str]:
        """Get list of pending peer emails."""
        self.client.load_peers(force_download=True)
        return [p.email for p in self.client.peer_manager.requested_by_peer_peers]

    def _get_domain(self, email: str) -> str:
        """Extract domain from email address."""
        return email.split("@")[-1].lower()

    def _should_approve(self, peer_email: str) -> tuple[bool, str]:
        """Check if a peer should be approved."""
        if self.state and self.state.was_approved(f"peer_{peer_email}"):
            return (False, "already approved by syft-approve")

        if peer_email in self.config.auto_approve_emails:
            return (True, "peer in auto_approve list")

        domain = self._get_domain(peer_email)
        if domain not in self.config.approved_domains:
            return (False, f"domain {domain} not in approved_domains")

        return (True, "ok")

    def _share_datasets(self, peer_email: str):
        """Share configured datasets with approved peer."""
        for dataset_name in self.config.auto_share_datasets:
            try:
                if hasattr(self.client, "share_dataset"):
                    self.client.share_dataset(dataset_name, peer_email)
                    if self.verbose:
                        print(f"   Shared dataset: {dataset_name}")
            except Exception as e:
                if self.verbose:
                    print(f"   Failed to share {dataset_name}: {e}")

    def check_and_approve(self) -> list[str]:
        """Check all pending peers and approve those matching criteria."""
        approved_peers = []

        for peer_email in self._get_pending_peers():
            should_approve, reason = self._should_approve(peer_email)

            if not should_approve:
                if self.verbose:
                    print(f"Skipped peer: {peer_email} ({reason})")
                continue

            try:
                self.client.approve_peer_request(peer_email, verbose=False)
                approved_peers.append(peer_email)

                if self.state:
                    domain = self._get_domain(peer_email)
                    self.state.mark_approved(f"peer_{peer_email}", domain)

                if self.verbose:
                    print(f"Approved peer: {peer_email}")

                if self.config.auto_share_datasets:
                    self._share_datasets(peer_email)

                if self.on_approve:
                    self.on_approve(peer_email)

            except Exception as e:
                if self.verbose:
                    print(f"Failed to approve peer {peer_email}: {e}")

        return approved_peers
