"""
Version negotiation module for syft-client.

Note: PeerManager is not exported here to avoid circular imports.
Import it directly: from syft_client.sync.version.peer_manager import PeerManager
"""

from syft_client.sync.version.version_info import VersionInfo
from syft_client.sync.version.exceptions import (
    VersionError,
    VersionMismatchError,
    VersionUnknownError,
    ClientVersionMismatchError,
    ProtocolVersionMismatchError,
)

__all__ = [
    "VersionInfo",
    "VersionError",
    "VersionMismatchError",
    "VersionUnknownError",
    "ClientVersionMismatchError",
    "ProtocolVersionMismatchError",
]
