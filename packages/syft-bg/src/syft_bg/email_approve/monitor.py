"""Pub/Sub-based monitor for Gmail reply notifications."""

import email.utils
import json
import queue
import threading
from google.cloud import pubsub_v1
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError

from syft_bg.common.state import JsonStateManager
from syft_bg.email_approve.gmail_watch import GmailWatcher
from syft_bg.email_approve.handler import EmailApproveHandler

# Check watch renewal every 15 minutes
WATCH_CHECK_INTERVAL = 15 * 60

# Backoff limits for subscriber reconnection
MIN_BACKOFF = 1
MAX_BACKOFF = 60

FLOW_CONTROL = pubsub_v1.types.FlowControl(
    max_messages=20,
    max_bytes=10 * 1024 * 1024,
)


class EmailApproveMonitor:
    """Monitors Gmail for reply-based job approvals via Pub/Sub."""

    def __init__(
        self,
        watcher: GmailWatcher,
        handler: EmailApproveHandler,
        state: JsonStateManager,
        credentials: Credentials,
        subscription_path: str,
        topic_name: str,
        do_email: str,
    ):
        self.watcher = watcher
        self.handler = handler
        self.state = state
        self.credentials = credentials
        self.subscription_path = subscription_path
        self.topic_name = topic_name
        self.do_email = do_email
        self._stop_event = threading.Event()
        self._history_queue: queue.Queue[tuple[str, object]] = queue.Queue()

    def start(self) -> threading.Thread:
        """Start the monitor in a background thread."""
        self._stop_event.clear()
        self._init_watch()

        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stop the monitor."""
        self._stop_event.set()

    def _init_watch(self):
        """Start Gmail watch and seed history ID."""
        stored = self.state.get_data("email_approve_last_history_id")
        history_id, _ = self.watcher.start_watch(self.topic_name)
        if not stored:
            self.state.set_data("email_approve_last_history_id", history_id)

    def _run(self):
        """Main run loop: subscribe to Pub/Sub with reconnect."""
        renew_thread = threading.Thread(target=self._watch_renew_loop, daemon=True)
        renew_thread.start()

        worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        worker_thread.start()

        backoff = MIN_BACKOFF
        subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)

        while not self._stop_event.is_set():
            try:
                print(
                    f"[EmailApproveMonitor] Starting Pub/Sub pull on "
                    f"{self.subscription_path}"
                )
                future = subscriber.subscribe(
                    self.subscription_path,
                    callback=self._on_pubsub_message,
                    flow_control=FLOW_CONTROL,
                )
                # this part is blocking
                future.result()
            except Exception as e:
                if self._stop_event.is_set():
                    break
                print(
                    f"[EmailApproveMonitor] Subscriber error: {e}. "
                    f"Reconnecting in {backoff}s"
                )
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                backoff = MIN_BACKOFF

        try:
            subscriber.close()
        except Exception:
            pass

        print("[EmailApproveMonitor] Stopped")

    def _on_pubsub_message(self, message):
        """Enqueue Pub/Sub notification for single-threaded processing."""
        try:
            payload = json.loads(message.data.decode("utf-8"))
            history_id = str(payload.get("historyId", ""))
        except Exception:
            message.ack()
            return

        if not history_id:
            message.ack()
            return

        self._history_queue.put((history_id, message))

    def _process_queue(self):
        """Single-threaded worker that processes history IDs from the queue."""
        while not self._stop_event.is_set():
            try:
                history_id, message = self._history_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self._process_history(history_id)
            except Exception as e:
                print(
                    f"[EmailApproveMonitor] Error processing history: {e}", flush=True
                )
            finally:
                message.ack()

    def _process_history(self, new_history_id: str):
        """Fetch and process new messages since last history ID."""
        last = self.state.get_data("email_approve_last_history_id")

        if new_history_id == last:
            print(f"[EmailApproveMonitor] historyId unchanged ({last}), skipping")
            return

        try:
            msg_ids, newest = self.watcher.list_history_message_ids(last)
        except HttpError as e:
            # Gmail returns 404 when the historyId is too old (history expires).
            # Reseed and move on; some messages may be missed.
            status = getattr(e.resp, "status", None)
            if status == 404:
                print(
                    f"[EmailApproveMonitor] Stale historyId, reseeding to "
                    f"{new_history_id}"
                )
                self.state.set_data("email_approve_last_history_id", new_history_id)
                return
            raise

        if len(msg_ids) > 0:
            print(f"[EmailApproveMonitor] Found {len(msg_ids)} message(s)")

        for msg_id in msg_ids:
            if self._stop_event.is_set():
                break
            self._process_message(msg_id)

        final = newest if int(newest) > int(new_history_id) else new_history_id
        self.state.set_data("email_approve_last_history_id", final)

    def _process_message(self, msg_id: str):
        """Check if a message is a reply to a job email and handle it."""
        try:
            msg = self.watcher.get_message(msg_id)
        except Exception as e:
            print(f"[EmailApproveMonitor] Failed to fetch message {msg_id}: {e}")
            return

        # Only process emails sent by the DO (replies from self)
        _, from_email = email.utils.parseaddr(msg.get_header("From") or "")
        if from_email.lower() != self.do_email.lower():
            print(
                f"[EmailApproveMonitor] Skipping msg {msg_id}: from={from_email}, expected={self.do_email}"
            )
            return

        if not msg.thread_id or not msg.reply_text:
            print(
                f"[EmailApproveMonitor] Missing thread_id or reply_text for message {msg_id}"
            )
            return

        try:
            print(
                f"[EmailApproveMonitor] Handling reply for thread {msg.thread_id}, reply_text: {msg.reply_text}"
            )
            self.handler.handle_reply(msg.thread_id, msg.reply_text)
        except Exception as e:
            print(
                f"[EmailApproveMonitor] Error handling reply for thread {msg.thread_id}: {e}"
            )

    def _watch_renew_loop(self):
        """Periodically renew the Gmail watch."""
        while not self._stop_event.is_set():
            self._stop_event.wait(WATCH_CHECK_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                self.watcher.renew_if_needed(self.topic_name)
            except Exception as e:
                print(f"[EmailApproveMonitor] Watch renewal failed: {e}")
