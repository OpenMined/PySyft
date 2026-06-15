"""Tests for the syft-bg notify Heartbeat thread."""

import threading
import time
from unittest.mock import MagicMock

from syft_bg.notify.gmail.sender import SendResult
from syft_bg.notify.heartbeat import Heartbeat


class TestHeartbeat:
    """Tests for Heartbeat lifecycle and emitted email contents."""

    def test_sends_immediately_then_periodically(self):
        """First send is at startup, then once per interval; stops cleanly."""
        sender = MagicMock()
        sender.notify_heartbeat.return_value = SendResult(success=True)

        stop_event = threading.Event()
        heartbeat = Heartbeat(
            sender=sender,
            do_email="do@test.com",
            interval=1,
            stop_event=stop_event,
        )

        thread = heartbeat.start()
        # Wait long enough for the startup send + at least one interval tick.
        time.sleep(2.5)
        heartbeat.stop()
        thread.join(timeout=2)

        assert not thread.is_alive(), "heartbeat thread did not stop"
        assert sender.notify_heartbeat.call_count >= 2

        first_call = sender.notify_heartbeat.call_args_list[0]
        assert first_call.kwargs["do_email"] == "do@test.com"
        assert first_call.kwargs["interval_seconds"] == 1

    def test_continues_on_send_failure(self):
        """A single failed send must not kill the heartbeat loop."""
        sender = MagicMock()
        sender.notify_heartbeat.side_effect = [
            Exception("boom"),
            SendResult(success=True),
            SendResult(success=True),
        ]

        heartbeat = Heartbeat(
            sender=sender,
            do_email="do@test.com",
            interval=1,
        )

        thread = heartbeat.start()
        time.sleep(2.5)
        heartbeat.stop()
        thread.join(timeout=2)

        assert sender.notify_heartbeat.call_count >= 2

    def test_subject_is_exact(self):
        """Sanity check: the subject string is the exact one we promised."""
        from syft_bg.notify.gmail.sender import GmailSender

        # Bypass __init__ so we don't need real credentials.
        sender = GmailSender.__new__(GmailSender)
        sender.use_html = False
        sender._renderer = None
        sender.send_email = MagicMock(
            return_value=SendResult(success=True, thread_id=None)
        )

        sender.notify_heartbeat(do_email="do@test.com", interval_seconds=86400)

        sender.send_email.assert_called_once()
        args = sender.send_email.call_args.args
        # send_email(to_email, subject, body_text, body_html)
        assert args[0] == "do@test.com"
        assert args[1] == "heartbeat: syft-bg still running, token is active"
        assert "syft-bg is still running" in args[2]
