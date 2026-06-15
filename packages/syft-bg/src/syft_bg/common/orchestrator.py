"""Base orchestrator for managing service monitors."""

import threading
import time
from typing import Literal, Optional

from syft_bg.common.monitor import Monitor

MonitorType = Literal["jobs", "peers"]


class BaseOrchestrator:
    """Base class for service orchestrators.

    Subclasses must implement:
    - __init__() to set up service-specific state
    - _init_monitors() to create job and peer monitors
    - from_config() classmethod to create instance from config file
    """

    def __init__(self):
        self._job_monitor: Optional[Monitor] = None
        self._peer_monitor: Optional[Monitor] = None
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self.interval: int = 30

    def setup(self) -> None:
        """Validate external dependencies. Override in subclasses.

        Raises on failure so the error message and traceback are captured.
        """

    @staticmethod
    def _wait_for_sync_ready(timeout: int = 120, label: str = "") -> None:
        """Block until the sync service has completed at least one cycle.

        The sync service touches a marker file after its first successful cycle;
        services that read or write the SyftBox cache must wait for that marker
        to avoid racing sync's initial event-replay storm.
        """
        # Local import: sync.orchestrator depends on common.orchestrator.
        from syft_bg.sync.orchestrator import sync_ready_path

        marker = sync_ready_path()
        waited = 0
        prefix = f"[{label}] " if label else ""
        while not marker.exists() and waited < timeout:
            if waited % 10 == 0:
                print(f"{prefix}Waiting for sync service...")
            time.sleep(1)
            waited += 1
        if waited >= timeout:
            print(f"{prefix}Timed out waiting for sync, starting anyway")

    def _init_monitors(self):
        """Initialize job and peer monitors. Override in subclass."""
        raise NotImplementedError("Subclass must implement _init_monitors()")

    def start(self, monitor_type: Optional[MonitorType] = None) -> "BaseOrchestrator":
        """Start monitors in background threads."""
        self._init_monitors()
        self._stop_event.clear()

        if monitor_type is None or monitor_type == "jobs":
            if self._job_monitor:
                thread = self._job_monitor.start(interval=self.interval)
                self._threads.append(thread)

        if monitor_type is None or monitor_type == "peers":
            if self._peer_monitor:
                thread = self._peer_monitor.start(interval=self.interval)
                self._threads.append(thread)

        return self

    def stop(self) -> None:
        """Stop all running monitors."""
        if self._job_monitor:
            self._job_monitor.stop()
        if self._peer_monitor:
            self._peer_monitor.stop()
        self._threads.clear()

    def run_once(self, monitor_type: Optional[MonitorType] = None) -> None:
        """Run a single check cycle."""
        self._init_monitors()

        if monitor_type is None or monitor_type == "jobs":
            if self._job_monitor:
                self._job_monitor._check_all_entities()

        if monitor_type is None or monitor_type == "peers":
            if self._peer_monitor:
                self._peer_monitor._check_all_entities()

    @property
    def is_running(self) -> bool:
        """Check if any monitor threads are running."""
        return any(t.is_alive() for t in self._threads)

    def run_loop(self, monitor_type: Optional[MonitorType] = None) -> None:
        """Run monitors in foreground (blocking)."""
        self._init_monitors()
        self._stop_event.clear()

        self._print_startup_info()

        try:
            while not self._stop_event.is_set():
                if monitor_type is None or monitor_type == "jobs":
                    if self._job_monitor:
                        try:
                            self._job_monitor._check_all_entities()
                        except Exception as e:
                            print(f"[JobMonitor] Error: {e}")

                if monitor_type is None or monitor_type == "peers":
                    if self._peer_monitor:
                        try:
                            self._peer_monitor._check_all_entities()
                        except Exception as e:
                            print(f"[PeerMonitor] Error: {e}")

                self._stop_event.wait(self.interval)

        except KeyboardInterrupt:
            print("\nShutting down...")

        self._stop_event.set()
        print("Daemon stopped")

    def _print_startup_info(self):
        """Print startup info. Override in subclass for custom output."""
        print(f"Starting {self.__class__.__name__}...")
        print(f"  Interval: {self.interval}s")
        print()
