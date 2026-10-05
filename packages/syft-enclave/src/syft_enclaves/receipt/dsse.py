"""Signing a receipt as a DSSE envelope.

DSSE (Dead Simple Signing Envelope) is the envelope Sigstore's Rekor log
accepts with a key of your own, and Rekor checks the signature itself before it
logs the entry. The key is the enclave's attested key, which the report in the
receipt's ``keyBinding`` lists (``receipt/key_binding.py``).
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from syft_enclaves.receipt.key_binding import KeyBindingError, bound_key

#: The payload is an in-toto statement, so it carries in-toto's payload type.
PAYLOAD_TYPE = "application/vnd.in-toto+json"


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


def sign_receipt(receipt: dict[str, Any], private_key: Ed25519PrivateKey) -> dict:
    """Wrap *receipt* in a DSSE envelope signed with *private_key*."""
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


def verify_receipt(envelope: dict[str, Any]) -> dict:
    """Check *envelope* was signed by the key its report lists; return the receipt."""
    if envelope.get("payloadType") != PAYLOAD_TYPE:
        raise ReceiptVerificationError(
            f"not a receipt: {envelope.get('payloadType')!r}"
        )
    payload = base64.b64decode(envelope["payload"])
    receipt = json.loads(payload)
    public_key = signing_key(receipt)
    _verify_any_signature(envelope.get("signatures") or [], public_key, payload)
    return receipt


def signing_key(receipt: dict[str, Any]) -> bytes:
    """The key *receipt* must be signed with: the one its report lists."""
    execution = receipt.get("predicate", {}).get("execution") or {}
    binding = (execution.get("attestation") or {}).get("keyBinding")
    if binding is None:
        raise ReceiptVerificationError("the receipt has no key binding")
    try:
        return bound_key(binding)
    except KeyBindingError as e:
        raise ReceiptVerificationError(f"key binding: {e}") from e


def _verify_any_signature(signatures: list, public_key: bytes, payload: bytes) -> None:
    verifier = Ed25519PublicKey.from_public_bytes(public_key)
    message = pae(PAYLOAD_TYPE, payload)
    for signature in signatures:
        try:
            verifier.verify(base64.b64decode(signature["sig"]), message)
            return
        except (InvalidSignature, KeyError, ValueError):
            continue
    raise ReceiptVerificationError("no signature matches the receipt key")


def public_key_pem(public_key: bytes) -> bytes:
    """A raw Ed25519 key as PEM, the form Rekor wants for a verifier."""
    return Ed25519PublicKey.from_public_bytes(public_key).public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
