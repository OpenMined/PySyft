"""Binding the receipt key to the hardware report, once per boot.

The receipt key is a Tinfoil attested key (``receipt/signing_key.py``), and
every v3 report lists it in its ``crypto_material``. The enclave asks Tinfoil
for one such report, with the sha256 of the key as its nonce, and puts it in
each receipt. That report says "this key was made inside an enclave running the
measured config", and the key signs every receipt of this boot, so one report
covers all of them.

The verifier here checks that the document lists the receipt's key under our
key id, and that its report data is derived from the nonce and those sections.
It does not check the hardware signature on the report: the tinfoil Python SDK
cannot appraise v3 documents yet, only tinfoil-go can.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import json
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from syft_enclaves.evidence.tinfoil import fetch_nonce_bound_document
from syft_enclaves.receipt.signing_key import ATTESTED_KEY_ID

#: How the nonce is derived from the receipt key.
NONCE_SCHEME = "sha256-run-public-key/1"
REPORT_DATA_V1 = "https://tinfoil.sh/report-data/v1"
CRYPTO_MATERIAL_V1 = "https://tinfoil.sh/crypto-material/v1"
KEY_SPKI_V1 = "https://tinfoil.sh/key/spki/v1"


class KeyBindingError(Exception):
    """The key binding does not name the receipt's key."""


def run_key_nonce(run_public_key: bytes) -> str:
    return hashlib.sha256(run_public_key).hexdigest()


@functools.lru_cache(maxsize=1)
def key_binding(run_public_key: bytes) -> dict[str, Any]:
    """A v3 report that lists *run_public_key* as our attested key.

    Cached, so it is fetched once per key, which is once per boot. Raises when
    Tinfoil cannot give one: a receipt nobody can tie to the enclave is no
    receipt, so the job ships ``receipt_error.txt`` instead.
    """
    try:
        document = fetch_nonce_bound_document(run_key_nonce(run_public_key))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Could not fetch the key binding report: {e}") from e
    binding = {
        "nonceScheme": NONCE_SCHEME,
        "keyId": ATTESTED_KEY_ID,
        "document": document,
    }
    check_key_binding(binding, run_public_key)
    return binding


def check_key_binding(binding: dict[str, Any], run_public_key: bytes) -> None:
    """Raise unless *binding*'s document lists *run_public_key* and is intact."""
    if binding.get("nonceScheme") != NONCE_SCHEME:
        raise KeyBindingError(f"unknown nonce scheme {binding.get('nonceScheme')!r}")
    document = binding.get("document") or {}
    challenge = document.get("challenge") or {}
    if challenge.get("nonce") != run_key_nonce(run_public_key):
        raise KeyBindingError("the report's nonce does not name the receipt key")
    if challenge.get("report_data") != _expected_report_data(document):
        raise KeyBindingError("the report data is not derived from the nonce")
    if attested_key(document, binding.get("keyId")) != run_public_key:
        raise KeyBindingError("the report does not list the receipt key")


def attested_key(document: dict[str, Any], key_id: Any) -> bytes:
    """The raw ed25519 key the document lists under *key_id*."""
    for item in _crypto_material(document):
        if item.get("id") == key_id and item.get("format") == KEY_SPKI_V1:
            return _raw_ed25519(item.get("data", ""))
    raise KeyBindingError(f"the report lists no attested key {key_id!r}")


def _crypto_material(document: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        section = json.loads(base64.b64decode(document["crypto_material"]))
    except (KeyError, ValueError, binascii.Error) as e:
        raise KeyBindingError(f"malformed crypto_material: {e}") from e
    if section.get("format") != CRYPTO_MATERIAL_V1:
        raise KeyBindingError(f"unknown crypto_material {section.get('format')!r}")
    return section.get("items") or []


def _raw_ed25519(spki_hex: str) -> bytes:
    try:
        key = serialization.load_der_public_key(bytes.fromhex(spki_hex))
    except ValueError as e:
        raise KeyBindingError(f"malformed attested key: {e}") from e
    if not isinstance(key, Ed25519PublicKey):
        raise KeyBindingError("the attested key is not an ed25519 key")
    return key.public_bytes_raw()


def _expected_report_data(document: dict[str, Any]) -> str:
    """REPORT_DATA per report-data/v1: sha256 over the label, nonce and sections."""
    challenge = document["challenge"]
    if challenge.get("report_data_algorithm") != REPORT_DATA_V1:
        raise KeyBindingError(
            f"unknown report data algorithm {challenge.get('report_data_algorithm')!r}"
        )
    try:
        digest = hashlib.sha256(REPORT_DATA_V1.encode())
        digest.update(bytes.fromhex(challenge["nonce"]))
        for section in ("crypto_material", "device_evidence"):
            digest.update(hashlib.sha256(base64.b64decode(document[section])).digest())
    except (KeyError, ValueError, binascii.Error) as e:
        raise KeyBindingError(f"malformed attestation document: {e}") from e
    return (digest.digest() + bytes(32)).hex()
