"""Gmail watch management and history fetching."""

import time
from typing import Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from syft_bg.email_approve.gmail_message import GmailMessage

# Renew watch 1 day before expiry
WATCH_RENEW_MARGIN_MS = 24 * 60 * 60 * 1000


class GmailWatcher:
    """Manages Gmail push notifications via Pub/Sub and history fetching.

    start_watch() tells Gmail to push to a Pub/Sub topic on new INBOX messages.
    This only handles the registration of the push notification,
    the subscription to the Pub/Sub topic is handled by the EmailApproveMonitor.
    The watch expires after max 7 days and must be renewed.
    """

    def __init__(self, credentials: Credentials):
        self._credentials = credentials
        self._service = build(
            "gmail", "v1", credentials=credentials, cache_discovery=False
        )
        self._watch_expiration_ms: Optional[int] = None

    def start_watch(self, topic_name: str) -> tuple[str, int]:
        """Start or renew Gmail watch on INBOX.

        Returns (history_id, expiration_ms).
        """
        body = {
            "topicName": topic_name,
        }
        # Register a Gmail push notification: any mailbox change -> Pub/Sub topic
        # No label filter so we catch both received and sent messages.
        resp = self._service.users().watch(userId="me", body=body).execute()
        history_id = str(resp["historyId"])
        expiration_ms = int(resp["expiration"])
        self._watch_expiration_ms = expiration_ms
        print(
            f"[GmailWatcher] Watch active: historyId={history_id} "
            f"expiration_ms={expiration_ms}"
        )
        return history_id, expiration_ms

    def renew_if_needed(self, topic_name: str) -> Optional[tuple[str, int]]:
        if self._watch_expiration_ms is None:
            return self.start_watch(topic_name)

        now_ms = int(time.time() * 1000)
        if now_ms >= self._watch_expiration_ms - WATCH_RENEW_MARGIN_MS:
            print("[GmailWatcher] Watch nearing expiration, renewing.")
            return self.start_watch(topic_name)

        return None

    def list_history_message_ids(self, start_history_id: str) -> tuple[set[str], str]:
        """Fetch message IDs added since start_history_id.

        Returns (message_ids, newest_history_id).
        """
        page_token = None
        message_ids: set[str] = set()
        newest_history_id = start_history_id

        while True:
            resp = (
                self._service.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=start_history_id,
                    historyTypes=["messageAdded"],
                    pageToken=page_token,
                )
                .execute()
            )

            if "historyId" in resp:
                newest_history_id = str(resp["historyId"])

            for item in resp.get("history", []):
                for added in item.get("messagesAdded", []):
                    msg = added.get("message", {})
                    mid = msg.get("id")
                    if mid:
                        message_ids.add(mid)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return message_ids, newest_history_id

    def get_message(self, msg_id: str) -> GmailMessage:
        """Fetch a full message by ID."""
        data = (
            self._service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )
        return GmailMessage(data)
