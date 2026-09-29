"""Binding the run key to the hardware report, once per boot.

The enclave asks Tinfoil for a report whose nonce is the sha256 of its run
public key. That report then says "this key was made inside an enclave running
the measured config", and the run key signs every receipt of this boot, so one
report covers all of them.

The verifier here checks that the document names the receipt's run key, and
that its report data is derived from that nonce. It does not check the
hardware signature on the report: the tinfoil Python SDK cannot appraise v3
documents yet, only tinfoil-go can.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import logging
from typing import Any, Optional

from syft_enclaves.evidence.tinfoil import fetch_nonce_bound_document

logger = logging.getLogger(__name__)

#: How the nonce is derived from the run key.
NONCE_SCHEME = "sha256-run-public-key/1"
REPORT_DATA_V1 = "https://tinfoil.sh/report-data/v1"


class KeyBindingError(Exception):
    """The key binding does not name the receipt's run key."""


def run_key_nonce(run_public_key: bytes) -> str:
    return hashlib.sha256(run_public_key).hexdigest()


@functools.lru_cache(maxsize=1)
def key_binding(run_public_key: bytes) -> Optional[dict[str, Any]]:
    """The report bound to *run_public_key*, or None when Tinfoil cannot give one.

    Cached, so it is fetched once per run key, which is once per boot. A failed
    fetch is not cached, so the next receipt tries again.
    """
    try:
        document = fetch_nonce_bound_document(run_key_nonce(run_public_key))
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Receipt goes out without a key binding: %s", e)
        return None
    return {"nonceScheme": NONCE_SCHEME, "document": document}


def check_key_binding(binding: dict[str, Any], run_public_key: bytes) -> None:
    """Raise unless *binding*'s document commits to *run_public_key*."""
    if binding.get("nonceScheme") != NONCE_SCHEME:
        raise KeyBindingError(f"unknown nonce scheme {binding.get('nonceScheme')!r}")
    challenge = (binding.get("document") or {}).get("challenge") or {}
    if challenge.get("nonce") != run_key_nonce(run_public_key):
        raise KeyBindingError("the report's nonce does not name the run key")
    if challenge.get("report_data") != _expected_report_data(binding["document"]):
        raise KeyBindingError("the report data is not derived from the nonce")


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
