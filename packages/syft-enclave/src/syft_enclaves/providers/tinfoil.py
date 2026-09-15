"""Attestation evidence on Tinfoil.

Tinfoil mounts a read-only ``/tinfoil`` directory into every container holding
the enclave's own attestation document, the verified config it booted with, and
the status of the launched containers. Collecting evidence is therefore a file
read — the enclave never verifies itself, and needs no Tinfoil SDK.

See https://docs.tinfoil.sh/containers/config-runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from syft_enclaves.attestation_envelope import (
    AttestationEvidence,
    AttestationKind,
    tinfoil_evidence,
)

TINFOIL_DIR = Path("/tinfoil")
TINFOIL_ATTESTATION_PATH = TINFOIL_DIR / "attestation.json"
TINFOIL_CONFIG_PATH = TINFOIL_DIR / "config.yml"
TINFOIL_STATUS_PATH = TINFOIL_DIR / "container-status.json"


class TinfoilProvider:
    """Evidence from Tinfoil's ``/tinfoil`` mount."""

    kind = AttestationKind.TINFOIL
    probe_path = TINFOIL_ATTESTATION_PATH
    #: Tinfoil cannot: the report's user data is the shim's own keys.
    accepts_caller_nonce = False

    def __init__(
        self,
        repo: Optional[str] = None,
        release_tag: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        # Recorded in the published evidence so a verifier can warn when its
        # own policy points at a different config repo. Never a source of trust.
        self.repo = repo
        self.release_tag = release_tag
        # Where peers can reach us for a pinned fetch. Also not trusted.
        self.host = host

    @classmethod
    def detect(cls) -> bool:
        return TINFOIL_ATTESTATION_PATH.exists()

    @classmethod
    def from_settings(cls, settings: Any = None) -> "TinfoilProvider":
        return cls(
            repo=getattr(settings, "tinfoil_repo", None),
            release_tag=getattr(settings, "tinfoil_release_tag", None),
            host=getattr(settings, "tinfoil_host", None),
        )

    def collect(self, caller_nonce: Optional[str] = None) -> AttestationEvidence:
        if caller_nonce is not None:
            raise ValueError(
                "Tinfoil evidence cannot carry a caller nonce: the report's 64 "
                "bytes of user data are fully used by the shim's TLS key "
                "fingerprint and HPKE public key, and the attestation document "
                "is a static file. Requesting a nonce here would silently give "
                "no freshness guarantee at all."
            )
        return tinfoil_evidence(
            self._read_attestation_document(),
            repo=self.repo,
            release_tag=self.release_tag,
            host=self.host,
        )

    def describe(self, evidence: AttestationEvidence) -> dict[str, Any]:
        return {
            "document": {"format": evidence.format, "body": evidence.body},
            "config": _read_text(TINFOIL_CONFIG_PATH),
            "container_status": _read_json(TINFOIL_STATUS_PATH),
            **evidence.metadata,
        }

    def _read_attestation_document(self) -> dict[str, Any]:
        try:
            document = json.loads(TINFOIL_ATTESTATION_PATH.read_text())
        except FileNotFoundError as e:
            raise RuntimeError(
                f"No Tinfoil attestation document at {TINFOIL_ATTESTATION_PATH}. "
                "This container is not running inside a Tinfoil enclave."
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Malformed Tinfoil attestation document at "
                f"{TINFOIL_ATTESTATION_PATH}: {e}"
            ) from e
        if not isinstance(document, dict):
            raise RuntimeError(
                f"Malformed Tinfoil attestation document at "
                f"{TINFOIL_ATTESTATION_PATH}: expected an object"
            )
        return document


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[Any]:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
