"""
Version-related exceptions for syft-client.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from syft_client.sync.version.version_info import VersionInfo


class VersionError(Exception):
    """Base exception for version-related errors."""

    pass


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


class VersionUnknownError(VersionError):
    """Raised when peer version information is not available."""

    def __init__(self, peer_email: str, operation: Optional[str] = None):
        self.peer_email = peer_email
        self.operation = operation

        if operation:
            message = (
                f"Cannot {operation} for peer {peer_email}: version information "
                "not available. The peer may be running an older version of "
                "syft-client that doesn't support version negotiation."
            )
        else:
            message = (
                f"Unknown version for peer {peer_email}. The peer may be running "
                "an older version of syft-client that doesn't support version negotiation."
            )

        super().__init__(message)


class ClientVersionMismatchError(VersionMismatchError):
    """Raised specifically for client version mismatches."""

    def __init__(
        self,
        peer_email: str,
        local_version: "VersionInfo",
        peer_version: "VersionInfo",
    ):
        reason = (
            f"Client version mismatch: local={local_version.syft_client_version}, "
            f"peer={peer_version.syft_client_version}"
        )
        super().__init__(peer_email, local_version, peer_version, reason)


class ProtocolVersionMismatchError(VersionMismatchError):
    """Raised specifically for protocol version mismatches."""

    def __init__(
        self,
        peer_email: str,
        local_version: "VersionInfo",
        peer_version: "VersionInfo",
    ):
        reason = (
            f"Protocol version mismatch: local={local_version.protocol_version}, "
            f"peer={peer_version.protocol_version}"
        )
        super().__init__(peer_email, local_version, peer_version, reason)
