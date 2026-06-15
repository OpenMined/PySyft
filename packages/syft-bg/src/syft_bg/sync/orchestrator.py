"""Sync orchestrator — runs the sync loop and writes snapshots."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from syft_bg.common.config import get_default_paths
from syft_bg.common.orchestrator import BaseOrchestrator
from syft_bg.common.state import JsonStateManager
from syft_bg.sync.config import SyncConfig
from syft_bg.sync.snapshot import PeerVersionInfo, SyncSnapshot

if TYPE_CHECKING:
    from syft_client.sync.syftbox_manager import SyftboxManager


def sync_ready_path() -> Path:
    """Marker file that sync writes once it has completed at least one cycle."""
    return get_default_paths().sync_state.parent / ".sync_ready"


class SyncOrchestrator(BaseOrchestrator):
    def __init__(
        self,
        client: SyftboxManager,
        state: JsonStateManager,
        config: SyncConfig,
    ):
        super().__init__()
        self.client = client
        self.state = state
        self.config = config
        self.interval = config.interval
        self._sync_count = 0

    @classmethod
    def from_config(cls, config: SyncConfig) -> SyncOrchestrator:
        if not config.do_email:
            raise ValueError("SyncConfig missing 'do_email'")
        if not config.syftbox_root:
            raise ValueError("SyncConfig missing 'syftbox_root'")

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

        state = JsonStateManager(config.sync_state_path)

        return cls(
            client=client,
            state=state,
            config=config,
        )

    def _init_monitors(self):
        """No-op: sync doesn't use monitors."""
        pass

    def setup(self) -> None:
        """Verify Drive credentials by running a single sync."""
        self.client.sync(force_download_peer_state=True)

    def run_loop(self, monitor_type=None) -> None:
        self._sync_ready_path().unlink(missing_ok=True)
        self._print_startup_info()
        try:
            while not self._stop_event.is_set():
                self._sync_and_snapshot()
                self._stop_event.wait(self.config.interval)
        except KeyboardInterrupt:
            print("\nShutting down...")
        self._stop_event.set()
        print("Daemon stopped")

    def run_once(self, monitor_type=None) -> None:
        self._sync_and_snapshot()

    def _sync_and_snapshot(self) -> None:
        start = time.time()
        sync_error = None

        try:
            self._sync_with_retry()
        except Exception as e:
            sync_error = str(e)

        snapshot = self._build_snapshot(start, sync_error)
        self.state.set_data("snapshot", snapshot.model_dump())

        duration_ms = snapshot.sync_duration_ms
        count = snapshot.sync_count
        if sync_error:
            print(
                f"[SyncOrchestrator] Cycle {count} failed in {duration_ms}ms: {sync_error}"
            )
        else:
            if self._sync_count == 1:
                marker = self._sync_ready_path()
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            job_count = len(snapshot.job_names)
            peer_count = len(snapshot.approved_peer_emails)
            print(
                f"[SyncOrchestrator] Cycle {count} completed in {duration_ms}ms "
                f"({job_count} jobs, {peer_count} peers)"
            )

    def _sync_with_retry(self) -> None:
        for attempt in range(self.config.max_retries):
            try:
                self.client.sync(force_download_peer_state=True)
                return
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                wait = self.config.retry_backoff**attempt
                print(
                    f"[SyncOrchestrator] Retry {attempt + 1}/{self.config.max_retries}: {e}"
                )
                time.sleep(wait)

    def _build_snapshot(
        self, start_time: float, sync_error: Optional[str]
    ) -> SyncSnapshot:
        job_names = []
        approved_peers = []
        all_peers = []
        own_version = None
        peer_versions = {}
        try:
            job_names = [j.name for j in self.client.job_client.jobs]
            approved_peers = [p.email for p in self.client.peer_manager.approved_peers]
            all_peers = approved_peers + [
                p.email for p in self.client.peer_manager.requested_by_peer_peers
            ]

            own_vi = self.client.peer_manager.get_own_version()
            own_version = PeerVersionInfo(
                syft_client_version=own_vi.syft_client_version,
                protocol_version=own_vi.protocol_version,
            )
            for peer_email in approved_peers:
                vi = self.client.peer_manager.get_peer_version(peer_email)
                if vi:
                    peer_versions[peer_email] = PeerVersionInfo(
                        syft_client_version=vi.syft_client_version,
                        protocol_version=vi.protocol_version,
                    )
        except Exception as e:
            print(f"[SyncOrchestrator] Error reading client state: {e}")

        self._sync_count += 1

        return SyncSnapshot(
            sync_time=time.time(),
            sync_count=self._sync_count,
            sync_error=sync_error,
            sync_duration_ms=int((time.time() - start_time) * 1000),
            job_names=job_names,
            peer_emails=all_peers,
            approved_peer_emails=approved_peers,
            own_version=own_version,
            peer_versions=peer_versions,
        )

    def _sync_ready_path(self) -> Path:
        return sync_ready_path()

    def _print_startup_info(self) -> None:
        print("Starting sync daemon...")
        print(f"  DO: {self.config.do_email}")
        print(f"  Interval: {self.config.interval}s")
        print(f"  Max retries: {self.config.max_retries}")
        print()
