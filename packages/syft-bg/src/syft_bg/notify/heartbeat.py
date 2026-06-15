"""Heartbeat thread for the notify service.

Periodically emails the data owner so they know the service is still running
and the Gmail token is still valid. Absence of the email is the alert signal.
"""

import threading
import traceback
from datetime import datetime
from typing import Optional

from syft_bg.notify.gmail.sender import GmailSender


class Heartbeat:
    """Sends a heartbeat email at startup and on a fixed interval."""

    def __init__(
        self,
        sender: GmailSender,
        do_email: str,
        interval: int,
        stop_event: Optional[threading.Event] = None,
    ):
        self.sender = sender
        self.do_email = do_email
        self.interval = interval
        self._stop_event = stop_event or threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> threading.Thread:
        """Start the heartbeat thread; returns the thread."""
        thread = threading.Thread(target=self._run, daemon=True, name="Heartbeat")
        self._thread = thread
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        print(f"[Heartbeat] Started (interval: {self.interval}s)")
        # Send one immediately, then loop on the interval.
        self._send_one()
        while not self._stop_event.is_set():
            interrupted = self._stop_event.wait(self.interval)
            if interrupted:
                break
            self._send_one()
        print("[Heartbeat] Stopped")

    def _send_one(self) -> None:
        try:
            result = self.sender.notify_heartbeat(
                do_email=self.do_email,
                interval_seconds=self.interval,
                timestamp=datetime.now(),
            )
            if result.success:
                print(
                    f"[Heartbeat] sent at {datetime.now().isoformat(timespec='seconds')}"
                )
            else:
                print(f"[Heartbeat] send failed: {result.error_message}")
        except Exception:
            print(f"[Heartbeat] send raised:\n{traceback.format_exc()}")
