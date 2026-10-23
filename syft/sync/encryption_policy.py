"""Encryption policy that a client advertises to its peers."""

from enum import Enum


class EncryptionPolicy(str, Enum):
    """What a client asks of the encryption between it and a peer.

    - ``OFF``: never encrypt.
    - ``OPTIONAL``: encrypt only with a peer that requires it.
    - ``REQUIRED``: encrypt with every peer, and refuse plaintext.

    A pair encrypts when one or both sides require it. An ``OFF`` and a
    ``REQUIRED`` client conflict, and do not exchange messages.
    """

    OFF = "off"
    OPTIONAL = "optional"
    REQUIRED = "required"

    @classmethod
    def coerce(cls, value: "bool | str | EncryptionPolicy") -> "EncryptionPolicy":
        """The policy for an ``encryption`` setting.

        ``True`` is ``REQUIRED`` and ``False`` is ``OPTIONAL``. A policy or
        its name (``"off"``) selects that policy.
        """
        if isinstance(value, bool):
            return cls.REQUIRED if value else cls.OPTIONAL
        return cls(value)


class EncryptionPolicyConflictError(ValueError):
    """One side of the pair has encryption off and the other requires it."""

    def __init__(self, peer_email: str, message: str | None = None):
        super().__init__(
            message
            or f"Encryption with {peer_email} is off on one side and required on "
            "the other, so this pair cannot exchange messages. Set the same "
            "policy on both sides, or use optional on one of them."
        )
        self.peer_email = peer_email


class PeerOutdatedError(EncryptionPolicyConflictError):
    """The peer runs a syft that predates encryption negotiation."""

    def __init__(self, peer_email: str):
        super().__init__(
            peer_email,
            f"{peer_email} runs an outdated syft: its version file has no "
            "encryption policy. Ask them to upgrade syft.",
        )


class MessageFormatError(ValueError):
    """A message is plaintext in a pair that encrypts, or the other way around."""

    def __init__(self, peer_email: str, encrypted: bool):
        if encrypted:
            detail = "is encrypted, and the pair does not encrypt"
        else:
            detail = "is not encrypted, and the pair encrypts"
        super().__init__(f"A message from {peer_email} {detail}.")
        self.peer_email = peer_email


def pair_uses_encryption(
    mine: EncryptionPolicy, theirs: EncryptionPolicy | None
) -> bool:
    """Whether a pair with these policies encrypts. ``theirs`` is None when unknown.

    A conflicting pair (see ``pair_conflicts``) does not encrypt.
    """
    if EncryptionPolicy.OFF in (mine, theirs):
        return False
    return EncryptionPolicy.REQUIRED in (mine, theirs)


def pair_conflicts(mine: EncryptionPolicy, theirs: EncryptionPolicy | None) -> bool:
    """Whether one side has encryption off and the other requires it."""
    return {mine, theirs} == {EncryptionPolicy.OFF, EncryptionPolicy.REQUIRED}
