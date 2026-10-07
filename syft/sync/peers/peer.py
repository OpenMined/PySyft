from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, PrivateAttr
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.peers.key_bundle import bundle_fingerprint
from syft.sync.platforms.base_platform import BasePlatform
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
    version: Optional[VersionInfo] = None
    public_encryption_bundle: Optional[dict] = None

    # Set by SyftboxManager.peers when this Peer is handed out.
    _manager: Any = PrivateAttr(default=None)
    # The policy of an earlier client, whose version file does not say.
    _legacy_policy: Optional[EncryptionPolicy] = PrivateAttr(default=None)

    @property
    def encryption_policy(self) -> Optional[EncryptionPolicy]:
        """The policy of the peer, or None while it is not known.

        It is not known before the version file is read, and for an earlier
        client until ``set_legacy_policy``.
        """
        if self.version is None:
            return None
        if self.version.encryption_policy is not None:
            return self.version.encryption_policy
        return self._legacy_policy

    @property
    def needs_legacy_policy(self) -> bool:
        """Whether the version file is read and does not say the policy."""
        return self.version is not None and self.version.encryption_policy is None

    def set_legacy_policy(self, policy: Optional[EncryptionPolicy]) -> None:
        """Set the policy of an earlier client, whose version file does not say.

        None keeps it unknown.
        """
        self._legacy_policy = policy

    @property
    def fingerprint(self) -> Optional[str]:
        """Fingerprint of the peer's pinned identity key, or None without a bundle.

        Compare it with the peer out of band: the bundle arrives over Drive, and
        only that comparison shows the key is theirs.
        """
        if self.public_encryption_bundle is None:
            return None
        try:
            return bundle_fingerprint(self.public_encryption_bundle)
        except ValueError:
            return None

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
