"""Encryption policy that a client advertises to its peers."""

from enum import Enum
from pathlib import Path


class EncryptionPolicy(str, Enum):
    """What a client asks of the encryption between it and a peer.

    - ``OPTIONAL``: encrypt only with a peer that requires it.
    - ``REQUIRED``: encrypt with every peer, and refuse plaintext.

    A pair encrypts when one or both sides require it.
    """

    OPTIONAL = "optional"
    REQUIRED = "required"

    @classmethod
    def from_bool(cls, encryption: bool | None) -> "EncryptionPolicy | None":
        """The policy for an ``encryption`` setting: ``True`` is ``REQUIRED``.

        ``None`` means no policy was asked for: the datasite keeps its policy.

        Raises:
            TypeError: if ``encryption`` is not a bool or None.
        """
        if encryption is None:
            return None
        if not isinstance(encryption, bool):
            raise TypeError(
                f"encryption must be True, False or None, got {encryption!r} "
                f"({type(encryption).__name__})"
            )
        return cls.REQUIRED if encryption else cls.OPTIONAL


class EncryptionPolicyMismatchError(ValueError):
    """A client asked for another policy than the datasite uses."""

    def __init__(
        self, path: Path, on_disk: EncryptionPolicy, requested: EncryptionPolicy
    ):
        super().__init__(
            f"This SyftBox uses encryption policy {on_disk.value!r} ({path}), "
            f"and this client asked for {requested.value!r}. Every process on "
            "a SyftBox uses the same policy. Log in without `encryption` to "
            f"use {on_disk.value!r}, or delete the SyftBox "
            "(client.delete_syftbox()) to start over with another policy."
        )
        self.on_disk = on_disk
        self.requested = requested


class MessageFormatError(ValueError):
    """A message is plaintext in a pair that encrypts, or the other way around."""

    def __init__(self, peer_email: str, encrypted: bool):
        if encrypted:
            detail = "is encrypted, and the pair does not encrypt"
        else:
            detail = "is not encrypted, and the pair encrypts"
        super().__init__(f"A message from {peer_email} {detail}.")
        self.peer_email = peer_email
