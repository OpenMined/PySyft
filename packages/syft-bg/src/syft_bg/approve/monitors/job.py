"""Job monitor: runs the jobs api calls became."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from syft_bg.approve.handlers.job import ApprovedJobHandler, StateManager
from syft_bg.common.monitor import Monitor

if TYPE_CHECKING:
    from syft_job.job import JobInfo
    from syft_rds import SyftRDSClient


class JobMonitor(Monitor):
    """Watches for approved jobs, and runs them."""

    def __init__(
        self,
        client: SyftRDSClient,
        config_path: Optional[Path] = None,
        state: Optional[StateManager] = None,
        on_approve: Optional[Callable[[JobInfo], None]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.handler = ApprovedJobHandler(
            client=client,
            config_path=config_path,
            state=state,
            on_approve=on_approve,
            verbose=verbose,
        )

    def _check_all_entities(self):
        """Run approved jobs."""
        self.handler.run_approved_jobs()
