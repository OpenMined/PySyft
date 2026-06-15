"""Peer event handler for notifications."""

from typing import Optional

from syft_bg.common.state import JsonStateManager
from syft_bg.notify.gmail.sender import GmailSender


class PeerHandler:
    """Handles peer-related notification events."""

    def __init__(
        self,
        sender: GmailSender,
        state: JsonStateManager,
        notify_on_new_peer: bool = True,
        notify_on_peer_granted: bool = True,
    ):
        self.sender = sender
        self.state = state
        self.notify_on_new_peer = notify_on_new_peer
        self.notify_on_peer_granted = notify_on_peer_granted

    def on_new_peer_request_to_do(
        self,
        do_email: str,
        ds_email: str,
        peer_url: Optional[str] = None,
    ) -> bool:
        """Notify DO about a new peer request."""
        if not self.notify_on_new_peer:
            return False

        state_key = f"peer_new_{ds_email}_to_do"
        if self.state.was_notified(state_key, "new_peer_request"):
            return False

        result = self.sender.notify_new_peer_request_to_do(do_email, ds_email, peer_url)

        if result.success:
            self.state.mark_notified(state_key, "new_peer_request")

        return result.success

    def on_peer_request_sent(
        self,
        ds_email: str,
        do_email: str,
    ) -> bool:
        """Notify DS that their peer request was sent."""
        if not self.notify_on_new_peer:
            return False

        state_key = f"peer_new_{ds_email}_to_ds"
        if self.state.was_notified(state_key, "peer_request_sent"):
            return False

        result = self.sender.notify_peer_request_sent(ds_email, do_email)

        if result.success:
            self.state.mark_notified(state_key, "peer_request_sent")

        return result.success

    def on_peer_added(
        self,
        ds_email: str,
        do_email: str,
    ) -> bool:
        """Notify DS that they added a peer."""
        if not self.notify_on_new_peer:
            return False

        state_key = f"peer_added_{ds_email}_{do_email}"
        if self.state.was_notified(state_key, "peer_added"):
            return False

        result = self.sender.notify_peer_added_to_ds(ds_email, do_email)

        if result.success:
            self.state.mark_notified(state_key, "peer_added")

        return result.success

    def on_peer_granted(
        self,
        ds_email: str,
        do_email: str,
    ) -> bool:
        """Notify DS that their peer request was accepted."""
        if not self.notify_on_peer_granted:
            return False

        state_key = f"peer_granted_{ds_email}"
        if self.state.was_notified(state_key, "peer_granted"):
            return False

        result = self.sender.notify_peer_request_granted(ds_email, do_email)

        if result.success:
            self.state.mark_notified(state_key, "peer_granted")

        return result.success
