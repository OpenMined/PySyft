"""Signing a receipt as a DSSE envelope, with the enclave's identity key.

DSSE (Dead Simple Signing Envelope) is the envelope Sigstore's Rekor log
accepts with a key of your own, and Rekor checks the signature itself before it
logs the entry. The key is the Ed25519 identity key from the enclave's syft
bundle — the one the attestation already binds (see ``attestation/nonce.py``),
so a verifier who attested the enclave already holds the key that checks the
receipt.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from syft_enclaves.attestation.nonce import identity_key_bytes, identity_private_key

PAYLOAD_TYPE = "application/vnd.openmined.syft-enclave-receipt+json"


class ReceiptVerificationError(Exception):
    """The receipt was not signed by the key it was checked against."""


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE's pre-authentication encoding: exactly the bytes that get signed."""
    kind = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(kind), kind, len(payload), payload)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def key_id(public_key: bytes) -> str:
    return hashlib.sha256(public_key).hexdigest()


def sign_receipt(receipt: dict[str, Any], private_jwks: dict[str, Any]) -> dict:
    """Wrap *receipt* in a DSSE envelope signed with the identity key."""
    private_key = identity_private_key(private_jwks)
    public_key = private_key.public_key().public_bytes_raw()
    payload = canonical_json(receipt)
    signature = private_key.sign(pae(PAYLOAD_TYPE, payload))
    return {
        "payloadType": PAYLOAD_TYPE,
        "payload": base64.b64encode(payload).decode(),
        "signatures": [
            {"keyid": key_id(public_key), "sig": base64.b64encode(signature).decode()}
        ],
    }


def verify_receipt(envelope: dict[str, Any], bundle: dict[str, Any]) -> dict:
    """Check *envelope* was signed by *bundle*'s identity key; return the receipt.

    Pass the bundle attestation verified (``result.verified_key_bundle``): that
    is what makes a valid signature mean "this enclave wrote it".
    """
    if envelope.get("payloadType") != PAYLOAD_TYPE:
        raise ReceiptVerificationError(f"not a receipt: {envelope.get('payloadType')!r}")
    public_key = identity_key_bytes(bundle)
    payload = base64.b64decode(envelope["payload"])
    _verify_any_signature(envelope.get("signatures") or [], public_key, payload)
    receipt = json.loads(payload)
    if receipt.get("execution", {}).get("runPublicKey") != public_key.hex():
        raise ReceiptVerificationError("runPublicKey does not match the signing key")
    return receipt


def _verify_any_signature(signatures: list, public_key: bytes, payload: bytes) -> None:
    verifier = Ed25519PublicKey.from_public_bytes(public_key)
    message = pae(PAYLOAD_TYPE, payload)
    for signature in signatures:
        try:
            verifier.verify(base64.b64decode(signature["sig"]), message)
            return
        except (InvalidSignature, KeyError, ValueError):
            continue
    raise ReceiptVerificationError("no signature matches the enclave's identity key")


def public_key_pem(bundle: dict[str, Any]) -> bytes:
    """The identity key as PEM, the form Rekor wants for a verifier."""
    public_key = Ed25519PublicKey.from_public_bytes(identity_key_bytes(bundle))
    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
