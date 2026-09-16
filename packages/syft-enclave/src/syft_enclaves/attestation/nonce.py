"""Proving an enclave holds the syft key it just handed us, right now.

The hardware report cannot carry a caller nonce: its 64 bytes of user data are
the shim's TLS key fingerprint and HPKE public key, and there is no parameter to
influence them. So freshness for the *syft* key comes from a challenge instead.

The client sends a random nonce; the enclave signs it with the Ed25519 identity
key from its own bundle; the client verifies that signature against the bundle
the pinned channel delivered. Together with the TLS pin that gives the full
chain:

    hardware report  →  commits to the shim's TLS key
    TLS pin          →  this channel really ends in that enclave
    key bundle       →  therefore authentic, it came down that channel
    nonce signature  →  and the enclave holds the private half, right now

Without the nonce a bundle would be authentic but unproven: nothing would show
the enclave could actually use the key we are about to encrypt to, and nothing
would tie the response to this exchange rather than an earlier one.
"""

from __future__ import annotations

import base64
import json
import secrets
from typing import Any, Optional

#: Domain separator, so a signature made here can never be replayed as a
#: signature for some other syft protocol that also signs with the identity key.
#: v2 covers the claims document as well as the nonce; a v1 enclave's signature
#: will simply fail to verify, which is the outcome we want from a mismatch.
CHALLENGE_PREFIX = b"syft-enclave-attestation-challenge-v2:"
NONCE_BYTES = 32


class NonceVerificationError(Exception):
    """The enclave did not prove possession of the key bundle it served."""


def new_nonce() -> str:
    """A fresh client-chosen nonce, hex encoded."""
    return secrets.token_hex(NONCE_BYTES)


def challenge_message(nonce: str, claims: Optional[dict[str, Any]] = None) -> bytes:
    """Exactly the bytes both sides sign and verify.

    Covers the claims document as well as the nonce, so a single signature
    proves three things at once: the enclave holds the key, the answer is for
    this exchange, and these are the facts it meant to assert. Canonical JSON
    so both sides hash the same bytes from the same values.
    """
    payload = json.dumps(
        {"nonce": nonce, "claims": claims}, sort_keys=True, separators=(",", ":")
    )
    return CHALLENGE_PREFIX + payload.encode()


def sign_challenge(
    private_jwks: dict[str, Any], nonce: str, claims: Optional[dict[str, Any]] = None
) -> str:
    """Sign *nonce* with the identity key from a JWKS, base64 encoded.

    Runs inside the enclave. Takes the JWKS rather than a ``PeerStore`` so the
    attestation HTTP server can sign without depending on the sync engine.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    identity = private_jwks["identity_key"]
    if identity.get("crv") != "Ed25519":
        raise ValueError(f"identity key is not Ed25519: {identity.get('crv')!r}")
    private_key = Ed25519PrivateKey.from_private_bytes(_b64url_decode(identity["d"]))
    message = challenge_message(nonce, claims)
    return base64.b64encode(private_key.sign(message)).decode()


def verify_challenge(
    bundle: dict[str, Any],
    nonce: str,
    signature_b64: str,
    claims: Optional[dict[str, Any]] = None,
) -> None:
    """Check the enclave signed *our* nonce with the bundle's identity key.

    Raises :class:`NonceVerificationError` on any failure — a bad signature is
    indistinguishable from a replayed or absent one, and all of them mean the
    same thing: no proof of possession.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not signature_b64:
        raise NonceVerificationError("the enclave returned no nonce signature")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(identity_key_bytes(bundle))
        message = challenge_message(nonce, claims)
        public_key.verify(base64.b64decode(signature_b64), message)
    except InvalidSignature as e:
        raise NonceVerificationError(
            "the nonce signature does not match the key bundle: the responder "
            "does not hold the private half of the key it served"
        ) from e
    except (ValueError, KeyError, TypeError) as e:
        raise NonceVerificationError(f"malformed nonce signature or bundle: {e}") from e


def identity_key_bytes(bundle: dict[str, Any]) -> bytes:
    """The bundle's Ed25519 identity public key, via syft-crypto's own parser."""
    import syft_crypto_python as syc

    parsed = syc.SyftPublicKeyBundle.from_did_document(bundle)
    key_bytes = parsed.identity_key_bytes
    return key_bytes() if callable(key_bytes) else key_bytes


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
