"""Encryption policy that a client advertises to its peers."""

from enum import Enum


class EncryptionPolicy(str, Enum):
    """What a client asks of the encryption between it and a peer.

    - ``OPTIONAL``: encrypt only with a peer that requires it.
    - ``REQUIRED``: encrypt with every peer, and refuse plaintext.

    A pair encrypts when one or both sides require it.
    """

    OPTIONAL = "optional"
    REQUIRED = "required"

    @classmethod
    def from_bool(cls, encryption: bool) -> "EncryptionPolicy":
        """The policy for an ``encryption`` setting: ``True`` is ``REQUIRED``.

        Raises:
            TypeError: if ``encryption`` is not a bool.
        """
        if not isinstance(encryption, bool):
            raise TypeError(
                f"encryption must be True or False, got {encryption!r} "
                f"({type(encryption).__name__})"
            )
        return cls.REQUIRED if encryption else cls.OPTIONAL


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
    """Whether a pair with these policies encrypts. ``theirs`` is None when unknown."""
    return EncryptionPolicy.REQUIRED in (mine, theirs)
