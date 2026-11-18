"""
Version-related exceptions for syft.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from syft.sync.version.version_info import VersionInfo


class VersionError(Exception):
    """Base exception for version-related errors."""


class VersionMismatchError(VersionError):
    """Raised when versions are incompatible between peers."""

    def __init__(
        self,
        peer_email: str,
        local_version: "VersionInfo",
        peer_version: "VersionInfo",
        reason: Optional[str] = None,
    ):
        self.peer_email = peer_email
        self.local_version = local_version
        self.peer_version = peer_version
        self.reason = reason

        message = f"Version mismatch with peer {peer_email}."
        if reason:
            message += f" {reason}"
        else:
            message += (
                f"\nLocal client version: {local_version.syft_client_version}"
                f"\nPeer client version: {peer_version.syft_client_version}"
                f"\nLocal protocol version: {local_version.protocol_version}"
                f"\nPeer protocol version: {peer_version.protocol_version}"
            )

        super().__init__(message)


def peer_info_unavailable_reason(peer_email: str) -> str:
    """Why a peer's info file cannot be read, and what to do about it.

    Shared by VersionUnknownError and the sync skip log so both say the same.
    """
    return (
        f"their account info is not available yet. Usually this means the "
        f"connection request with {peer_email} has not been approved yet, or "
        f"{peer_email} has not created their account yet, or recently deleted it. "
        "Once that is fixed, call client.sync() and try again"
    )


class VersionUnknownError(VersionError):
    """Raised when a peer's info file (version and more) is not available."""

    def __init__(self, peer_email: str, operation: Optional[str] = None):
        self.peer_email = peer_email
        self.operation = operation

        reason = peer_info_unavailable_reason(peer_email)
        if operation:
            message = f"Cannot {operation} for peer {peer_email}: {reason}."
        else:
            message = f"Peer {peer_email}: {reason}."

        super().__init__(message)
