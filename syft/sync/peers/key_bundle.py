"""Validation and fingerprints for peer public key bundles.

A peer's public encryption bundle is a DID document that arrives over Google
Drive, and the threat model treats that transport as adversarial. Before a
bundle is pinned it must:

- parse as a DID document whose inner signatures verify: the identity key
  signed the key-agreement keys it is bundled with, and
- assert the identity it is filed under: ``id`` is ``did:syft:<email>`` and the
  ``identity`` field, when present, is that email.

Neither check proves the identity key belongs to the person. Only comparing
fingerprints out of band does, which is what :func:`format_fingerprint` is for.
"""

from typing import Optional

import syft_crypto_python as syc

DID_PREFIX = "did:syft:"


class InvalidPeerBundleError(ValueError):
    """The bundle does not parse, its signatures fail, or it names another identity."""


class PeerKeyChangedError(ValueError):
    """A bundle for an already-pinned peer carries a different identity key."""

    def __init__(
        self, peer_email: str, old_fingerprint: str, new_fingerprint: str
    ) -> None:
        self.peer_email = peer_email
        self.old_fingerprint = old_fingerprint
        self.new_fingerprint = new_fingerprint
        super().__init__(
            f"The encryption key of {peer_email} changed.\n"
            f"  pinned: {format_fingerprint(old_fingerprint)}\n"
            f"  new:    {format_fingerprint(new_fingerprint)}\n"
            "A key changes when the peer reinstalls or regenerates their keys, "
            "and also when someone tampers with the key exchange. Confirm the "
            "new fingerprint with the peer out of band before trusting it."
        )


class PeerFingerprintMismatchError(ValueError):
    """The bundle a peer publishes does not carry the fingerprint they confirmed."""

    def __init__(
        self, peer_email: str, expected_fingerprint: str, published_fingerprint: str
    ) -> None:
        self.peer_email = peer_email
        self.expected_fingerprint = expected_fingerprint
        self.published_fingerprint = published_fingerprint
        super().__init__(
            f"The key {peer_email} publishes does not match the fingerprint you "
            "confirmed with them.\n"
            f"  confirmed: {format_fingerprint(expected_fingerprint)}\n"
            f"  published: {format_fingerprint(published_fingerprint)}\n"
            "The published key was not trusted. Someone may be tampering with the "
            "key exchange, or the peer published a newer key since you compared."
        )


def did_for_email(email: str) -> str:
    return f"{DID_PREFIX}{email}"


def _same_email(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def format_fingerprint(fingerprint: Optional[str]) -> str:
    """Group a hex fingerprint into blocks of four for reading aloud."""
    if not fingerprint:
        return ""
    return " ".join(fingerprint[i : i + 4] for i in range(0, len(fingerprint), 4))


def normalize_fingerprint(fingerprint: str) -> str:
    """Undo :func:`format_fingerprint`, so a pasted fingerprint compares equal."""
    return "".join(fingerprint.split()).lower()


def bundle_fingerprint(bundle: dict) -> str:
    """Fingerprint of the identity key in ``bundle`` (no identity check)."""
    return _parse(bundle).identity_fingerprint()


def _parse(bundle: dict) -> syc.SyftPublicKeyBundle:
    if not isinstance(bundle, dict):
        raise InvalidPeerBundleError(
            f"Expected a DID document (dict), got {type(bundle).__name__}"
        )
    # from_did_document raises ValueError for a document it cannot read. Any
    # other error is a bug, and it must not pass as "invalid bundle": that makes
    # load_keys drop every pin and the next sync trust whatever Drive serves.
    try:
        parsed = syc.SyftPublicKeyBundle.from_did_document(bundle)
    except ValueError as e:
        raise InvalidPeerBundleError(f"Could not parse key bundle: {e}") from e
    if not parsed.verify_signatures():
        raise InvalidPeerBundleError("Key bundle signatures do not verify")
    return parsed


def parse_and_validate_bundle(peer_email: str, bundle: dict) -> syc.SyftPublicKeyBundle:
    """Parse ``bundle``, verify its signatures, and check it belongs to ``peer_email``.

    Raises:
        InvalidPeerBundleError: on any failure.
    """
    parsed = _parse(bundle)

    did = bundle.get("id")
    if not isinstance(did, str) or not did.startswith(DID_PREFIX):
        raise InvalidPeerBundleError(
            f"Key bundle for {peer_email} has no {DID_PREFIX!r} id (got {did!r})"
        )
    did_email = did[len(DID_PREFIX) :]
    if not _same_email(did_email, peer_email):
        raise InvalidPeerBundleError(
            f"Key bundle filed under {peer_email} asserts the identity {did_email!r}"
        )

    identity = bundle.get("identity")
    if identity is not None and (
        not isinstance(identity, str) or not _same_email(identity, peer_email)
    ):
        raise InvalidPeerBundleError(
            f"Key bundle filed under {peer_email} carries identity {identity!r}"
        )

    return parsed
