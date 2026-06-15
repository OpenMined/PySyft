"""Peer monitor for auto-approval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from syft_bg.approve.handlers.peer import PeerApprovalHandler, StateManager
from syft_bg.common.monitor import Monitor

if TYPE_CHECKING:
    from syft_bg.approve.config import PeerApprovalConfig
    from syft_client.sync.syftbox_manager import SyftboxManager


class PeerMonitor(Monitor):
    """Monitors for peers to auto-approve."""

    def __init__(
        self,
        client: SyftboxManager,
        config: PeerApprovalConfig,
        state: Optional[StateManager] = None,
        on_approve: Optional[Callable[[str], None]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.handler = PeerApprovalHandler(
            client=client,
            config=config,
            state=state,
            on_approve=on_approve,
            verbose=verbose,
        )

    def _check_all_entities(self):
        """Check and approve matching peers."""
        self.handler.check_and_approve()
