"""Job monitor for auto-approval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from syft_bg.approve.handlers.job import JobApprovalHandler, StateManager
from syft_bg.common.monitor import Monitor

if TYPE_CHECKING:
    from syft_job.job import JobInfo
    from syft_client.sync.syftbox_manager import SyftboxManager


class JobMonitor(Monitor):
    """Monitors for jobs to auto-approve."""

    def __init__(
        self,
        client: SyftboxManager,
        config_path: Optional[Path] = None,
        state: Optional[StateManager] = None,
        on_approve: Optional[Callable[[JobInfo], None]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.handler = JobApprovalHandler(
            client=client,
            config_path=config_path,
            state=state,
            on_approve=on_approve,
            verbose=verbose,
        )

    def _check_all_entities(self):
        """Check and approve matching jobs."""
        self.handler.check_and_approve()
