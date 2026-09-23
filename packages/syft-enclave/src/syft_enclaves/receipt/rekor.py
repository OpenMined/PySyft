"""Logging a signed receipt on Rekor, Sigstore's public transparency log.

The entry is a ``dsse`` entry. Rekor checks the envelope's signature against
the key we hand it, then stores hashes of the payload and the envelope, the
signature and the key — not the payload. So the log proves *that* the enclave
signed this exact receipt, and when, without publishing what is in it.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

from syft_enclaves.receipt.dsse import public_key_pem

REKOR_URL = "https://rekor.sigstore.dev"
SEARCH_URL = "https://search.sigstore.dev/?logIndex={}"


def upload_to_rekor(
    envelope: dict[str, Any], bundle: dict[str, Any], rekor_url: str = REKOR_URL
) -> dict[str, Any]:
    """Log *envelope*, signed by *bundle*'s identity key. Returns the entry.

    Uploading the same envelope twice is not an error: Rekor answers 409 with
    the existing entry, and that entry is returned.
    """
    request = urllib.request.Request(
        f"{rekor_url}/api/v1/log/entries",
        data=json.dumps(_dsse_entry(envelope, bundle)).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return _summarise(json.load(response))
    except urllib.error.HTTPError as e:
        if e.code != 409 or not e.headers.get("Location"):
            raise RuntimeError(f"Rekor rejected the receipt: {e.read().decode()}") from e
        return _fetch_existing(rekor_url, e.headers["Location"])


def _dsse_entry(envelope: dict[str, Any], bundle: dict[str, Any]) -> dict:
    verifier = base64.b64encode(public_key_pem(bundle)).decode()
    return {
        "apiVersion": "0.0.1",
        "kind": "dsse",
        "spec": {
            "proposedContent": {
                "envelope": json.dumps(envelope),
                "verifiers": [verifier],
            }
        },
    }


def _fetch_existing(rekor_url: str, location: str) -> dict[str, Any]:
    url = location if location.startswith("http") else f"{rekor_url}{location}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return _summarise(json.load(response))


def _summarise(entries: dict[str, Any]) -> dict[str, Any]:
    uuid, entry = next(iter(entries.items()))
    return {
        "uuid": uuid,
        "logIndex": entry["logIndex"],
        "integratedTime": entry.get("integratedTime"),
        "url": SEARCH_URL.format(entry["logIndex"]),
    }
