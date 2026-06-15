"""Notification orchestrator for email notifications."""

from typing import Optional

from syft_bg.common.orchestrator import BaseOrchestrator, MonitorType
from syft_bg.common.state import JsonStateManager
from syft_bg.notify.config import NotifyConfig
from syft_bg.notify.gmail.auth import GmailAuth
from syft_bg.notify.gmail.sender import GmailSender
from syft_bg.notify.handlers.job import JobHandler
from syft_bg.notify.handlers.peer import PeerHandler
from syft_bg.notify.heartbeat import Heartbeat
from syft_bg.notify.monitors.job import JobMonitor
from syft_bg.notify.monitors.peer import PeerMonitor


class NotificationOrchestrator(BaseOrchestrator):
    """Orchestrator for email notification service."""

    def __init__(
        self,
        config: NotifyConfig,
        job_monitor: JobMonitor,
        peer_monitor: Optional[PeerMonitor] = None,
        heartbeat: Optional[Heartbeat] = None,
    ):
        super().__init__()
        self.config = config
        self.interval = config.interval
        self._job_monitor = job_monitor
        self._peer_monitor: Optional[PeerMonitor] = peer_monitor
        self._heartbeat: Optional[Heartbeat] = heartbeat

    def _init_monitors(self):
        """No-op: monitors are created in from_config."""
        pass

    def setup(self) -> None:
        self._wait_for_sync_ready(label="Notify")
        self._job_monitor.handler.sender.verify()
        if self._job_monitor.state.is_empty():
            self._job_monitor.seed_existing_jobs()
            if self._peer_monitor:
                self._seed_existing_peers()

    @classmethod
    def from_config(
        cls,
        config: NotifyConfig,
    ) -> "NotificationOrchestrator":
        """Create orchestrator from a NotifyConfig."""
        if not config.do_email:
            raise ValueError("Config missing 'do_email' field")
        if not config.syftbox_root:
            raise ValueError("Config missing 'syftbox_root' field")

        if not config.gmail_token_path or not config.gmail_token_path.exists():
            raise FileNotFoundError(
                f"Gmail token not found: {config.gmail_token_path}\n\n"
                "Run setup to configure Gmail authentication first."
            )
        credentials = GmailAuth().load_credentials(config.gmail_token_path)
        sender = GmailSender(credentials)

        state_manager = JsonStateManager(config.notify_state_path)

        job_handler = JobHandler(
            sender,
            state_manager,
            do_email=config.do_email,
            syftbox_root=config.syftbox_root,
        )
        peer_handler = PeerHandler(sender, state_manager)

        job_monitor = JobMonitor(
            syftbox_root=config.syftbox_root,
            do_email=config.do_email,
            handler=job_handler,
            state=state_manager,
        )

        sync_state = JsonStateManager(config.sync_state_path)

        peer_monitor = PeerMonitor(
            do_email=config.do_email,
            handler=peer_handler,
            state=state_manager,
            sync_state=sync_state,
        )

        heartbeat: Optional[Heartbeat] = None
        if config.heartbeat_enabled:
            heartbeat = Heartbeat(
                sender=sender,
                do_email=config.do_email,
                interval=config.heartbeat_interval,
            )

        return cls(
            config=config,
            job_monitor=job_monitor,
            peer_monitor=peer_monitor,
            heartbeat=heartbeat,
        )

    def notify_peer_granted(self, ds_email: str) -> bool:
        """Notify DS that their peer request was granted."""
        if self._peer_monitor:
            return self._peer_monitor.notify_peer_granted(ds_email)
        return False

    def _seed_existing_peers(self):
        snapshot = self._peer_monitor._read_snapshot()
        if not snapshot or not snapshot.peer_emails:
            return
        state = self._peer_monitor.state
        state.set_data("peer_snapshot", snapshot.peer_emails)
        for peer_email in snapshot.approved_peer_emails:
            state_key = f"peer_granted_{peer_email}"
            state.mark_notified(state_key, "peer_granted")
        print(
            f"[PeerMonitor] Seeded {len(snapshot.peer_emails)} existing peers on fresh state"
        )

    def start(self, monitor_type: Optional[MonitorType] = None) -> "BaseOrchestrator":
        result = super().start(monitor_type)
        self._start_heartbeat()
        return result

    def run_loop(self, monitor_type: Optional[MonitorType] = None) -> None:
        # Share the orchestrator's stop event with heartbeat so a single
        # KeyboardInterrupt / stop() takes the heartbeat down too.
        self._stop_event.clear()
        self._start_heartbeat()
        super().run_loop(monitor_type)

    def stop(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.stop()
        super().stop()

    def _start_heartbeat(self) -> None:
        if self._heartbeat is None:
            return
        # Wire the orchestrator's shared stop_event into heartbeat so stop()
        # / run_loop() exit interrupts the heartbeat sleep.
        self._heartbeat._stop_event = self._stop_event
        thread = self._heartbeat.start()
        self._threads.append(thread)

    def _print_startup_info(self):
        """Print startup info for notify service."""
        print("Starting notification daemon...")
        print(f"  DO: {self.config.do_email}")
        print(f"  SyftBox: {self.config.syftbox_root}")
        print(f"  Interval: {self.config.interval}s")
        if self.config.heartbeat_enabled:
            print(f"  Heartbeat: every {self.config.heartbeat_interval}s")
        print()
