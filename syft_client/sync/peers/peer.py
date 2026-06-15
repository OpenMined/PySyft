from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, PrivateAttr
from syft_client.sync.platforms.base_platform import BasePlatform
from syft_client.sync.version.version_info import VersionInfo


class PeerState(str, Enum):
    ACCEPTED = "accepted"
    REQUESTED_BY_PEER = "requested_by_peer"
    REJECTED = "rejected"
    REQUESTED_BY_ME = "requested_by_me"


class Peer(BaseModel):
    email: str
    platforms: List[BasePlatform] = []
    state: PeerState = PeerState.ACCEPTED  # Default for backward compatibility
    version: Optional[VersionInfo] = None
    public_encryption_bundle: Optional[dict] = None
    use_encryption: bool = False

    # Set by SyftboxManager.peers when this Peer is handed out.
    _manager: Any = PrivateAttr(default=None)

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
