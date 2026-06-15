"""Gmail authentication and sending."""

from syft_bg.notify.gmail.auth import GmailAuth
from syft_bg.notify.gmail.sender import GmailSender

__all__ = ["GmailAuth", "GmailSender"]
