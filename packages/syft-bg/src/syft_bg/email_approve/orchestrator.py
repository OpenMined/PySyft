"""Orchestrator for email-based job approval service."""

from __future__ import annotations

from typing import Optional

from syft_job import SyftJobConfig
from syft_job.client import JobClient
from syft_job.job_runner import SyftJobRunner

from syft_bg.common.orchestrator import BaseOrchestrator
from syft_bg.common.state import JsonStateManager
from syft_bg.email_approve.config import EmailApproveConfig
from syft_bg.email_approve.gmail_watch import GmailWatcher
from syft_bg.email_approve.handler import EmailApproveHandler
from syft_bg.email_approve.monitor import EmailApproveMonitor
from syft_bg.email_approve.pubsub_setup import setup_pubsub
from syft_bg.notify.gmail.auth import GmailAuth


class EmailApproveOrchestrator(BaseOrchestrator):
    """Orchestrator for email-based job approval via Gmail push notifications."""

    def __init__(
        self,
        config: EmailApproveConfig,
        monitor: Optional[EmailApproveMonitor] = None,
    ):
        super().__init__()
        self.config = config
        self._monitor: Optional[EmailApproveMonitor] = monitor

    def _init_monitors(self):
        """No-op: push-based service uses EmailApproveMonitor directly."""
        pass

    def setup(self) -> None:
        """Verify Gmail credentials and Pub/Sub are accessible."""
        profile = (
            self._monitor.watcher._service.users().getProfile(userId="me").execute()
        )
        if self.config.gcp_project_id is None:
            raise ValueError(
                "GCP project ID not found, please set it in ~/.syft-bg/config.yaml"
            )
        print(f"[EmailApproveOrchestrator] Gmail OK: {profile.get('emailAddress')}")

    @classmethod
    def from_config(
        cls,
        config: EmailApproveConfig,
    ) -> EmailApproveOrchestrator:
        """Create orchestrator from an EmailApproveConfig."""
        if not config.do_email:
            raise ValueError("Config missing 'do_email' field")
        if not config.syftbox_root:
            raise ValueError("Config missing 'syftbox_root' field")

        if not config.gmail_token_path.exists():
            raise FileNotFoundError(
                f"Gmail token not found: {config.gmail_token_path}\n"
                "Run 'syft-bg init' first."
            )

        # Local-only job infrastructure (no Drive token needed)
        job_config = SyftJobConfig(
            syftbox_folder=config.syftbox_root,
            current_user_email=config.do_email,
        )
        job_client = JobClient.from_config(job_config)
        job_runner = SyftJobRunner.from_config(job_config)

        credentials = GmailAuth().load_credentials(config.gmail_token_path)

        watcher = GmailWatcher(credentials)
        state = JsonStateManager(config.email_approve_state_path)
        notify_state = JsonStateManager(config.notify_state_path)

        handler = EmailApproveHandler(
            job_client=job_client,
            job_runner=job_runner,
            state=state,
            notify_state=notify_state,
            do_email=config.do_email,
        )

        # Auto-create Pub/Sub resources if needed
        if not config.pubsub_topic or not config.pubsub_subscription:
            topic_path, sub_path = setup_pubsub(credentials, config.gcp_project_id)
            config.pubsub_topic = topic_path
            config.pubsub_subscription = sub_path
            config.save_pubsub_config()

        monitor = EmailApproveMonitor(
            watcher=watcher,
            handler=handler,
            state=state,
            credentials=credentials,
            subscription_path=config.pubsub_subscription,
            topic_name=config.pubsub_topic,
            do_email=config.do_email,
        )

        return cls(config=config, monitor=monitor)

    def run_loop(self) -> None:
        """Run the email approval service (blocking)."""
        assert self._monitor is not None

        print("Starting email approval daemon...")
        print(f"  DO: {self.config.do_email}")
        print(f"  Topic: {self.config.pubsub_topic}")
        print(f"  Subscription: {self.config.pubsub_subscription}")
        print()

        thread = self._monitor.start()
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._monitor.stop()

    def run_once(self) -> None:
        """Not meaningful for push-based service."""
        print(
            "[EmailApproveOrchestrator] Push-based service does not support "
            "--once. Use run_loop instead."
        )

    def stop(self) -> None:
        """Stop the service."""
        if self._monitor:
            self._monitor.stop()
