from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.platforms.base_platform import BasePlatform
from syft.sync.version.peer_info import PeerInfo
from syft.sync.version.version_info import VersionInfo


class PeerSetupError(ValueError):
    """A peer is unknown, not approved, or not shown to be a live connection."""


class PeerNotReadyError(PeerSetupError):
    """The peer can still become valid: it did not approve yet, or log in recently."""


class PeerState(str, Enum):
    ACCEPTED = "accepted"
    REQUESTED_BY_PEER = "requested_by_peer"
    REJECTED = "rejected"
    REQUESTED_BY_ME = "requested_by_me"


class Peer(BaseModel):
    email: str
    platforms: List[BasePlatform] = []
    state: PeerState = PeerState.ACCEPTED  # Default for backward compatibility
    info: Optional[PeerInfo] = None
    public_encryption_bundle: Optional[dict] = None
    # The state at the first load_peers() of this process, for a peer that was
    # connected or waiting then. Known in this process only, so not saved.
    state_at_start: Optional[PeerState] = Field(default=None, exclude=True)

    # Set by SyftboxManager.peers when this Peer is handed out.
    _manager: Any = PrivateAttr(default=None)

    @property
    def version(self) -> Optional[VersionInfo]:
        """The version the peer advertises, or None before its version file is read."""
        return self.info.version_info if self.info else None

    @property
    def encryption_policy(self) -> Optional[EncryptionPolicy]:
        """The policy the peer advertises, or None before its version file is read."""
        return self.info.encryption_policy if self.info else None

    @property
    def is_approved(self) -> bool:
        """Returns True if peer is accepted"""
        return self.state == PeerState.ACCEPTED

    @property
    def is_requested_by_peer(self) -> bool:
        """Returns True if peer requested us but we haven't added them yet"""
        return self.state == PeerState.REQUESTED_BY_PEER

    @property
    def is_requested_by_me(self) -> bool:
        """Returns True if we requested peer but they haven't reciprocated"""
        return self.state == PeerState.REQUESTED_BY_ME
