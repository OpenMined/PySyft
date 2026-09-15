"""Attestation evidence on GCP Confidential Space.

The launcher issues a Google-signed OIDC token over a Unix socket. A workload
can inject up to two nonces into it, which is the one channel a Confidential
Space workload has for binding its own data into hardware-signed evidence.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any, Optional

from syft_enclaves.attestation_envelope import (
    AttestationEvidence,
    AttestationKind,
    confidential_space_evidence,
)
from syft_enclaves.tee_token import (
    TEE_SOCKET_PATH,
    TOKEN_AUDIENCE,
    build_eat_nonce,
    fetch_attestation_token,
)


class ConfidentialSpaceProvider:
    """Evidence from the Confidential Space launcher."""

    kind = AttestationKind.CONFIDENTIAL_SPACE
    probe_path = TEE_SOCKET_PATH
    #: Confidential Space lets a workload inject nonces into the token.
    accepts_caller_nonce = True

    @classmethod
    def detect(cls) -> bool:
        return TEE_SOCKET_PATH.exists()

    @classmethod
    def from_settings(cls, settings: Any = None) -> "ConfidentialSpaceProvider":
        # Nothing to configure: the launcher socket and audience are fixed.
        return cls()

    def collect(self, caller_nonce: Optional[str] = None) -> AttestationEvidence:
        token = fetch_attestation_token(eat_nonce=build_eat_nonce(caller_nonce))
        return confidential_space_evidence(token, audience=TOKEN_AUDIENCE)

    def describe(self, evidence: AttestationEvidence) -> dict[str, Any]:
        return structure_claims(decode_jwt_payload(evidence.body))


def decode_jwt_payload(token: str) -> dict[str, Any]:
    """Base64-decode a JWT's payload segment **without** verifying the signature.

    For display only. A relying party verifies the token itself against
    Google's JWKS; see ``syft_enclaves.attestation``.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid JWT: expected 3 parts, got {len(parts)}")

    payload_b64 = parts[1]
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding
    return json.loads(base64.urlsafe_b64decode(payload_b64))


def _format_timestamp(epoch: int | float | None) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def structure_claims(claims: dict[str, Any]) -> dict[str, Any]:
    """Organize raw JWT claims into logical sections for display."""
    submods = claims.get("submods", {})
    container = submods.get("container", {})
    gce = submods.get("gce", {})

    result: dict[str, Any] = {
        "hardware": {
            "hwmodel": claims.get("hwmodel"),
            "secboot": claims.get("secboot"),
            "dbgstat": claims.get("dbgstat"),
        },
        "software": {
            "swname": claims.get("swname"),
            "swversion": claims.get("swversion"),
        },
        "container": {
            "image_digest": container.get("image_digest"),
            "image_reference": container.get("image_reference"),
            "restart_policy": container.get("restart_policy"),
            "env": container.get("env"),
        },
        "gce": {
            "project_id": gce.get("project_id"),
            "zone": gce.get("zone"),
            "instance_id": gce.get("instance_id"),
        },
        "issuer": claims.get("iss"),
        "subject": claims.get("sub"),
        "issued_at": _format_timestamp(claims.get("iat")),
        "expires_at": _format_timestamp(claims.get("exp")),
    }
    result.update(_optional_claim_sections(claims, submods))
    return result


def _optional_claim_sections(
    claims: dict[str, Any], submods: dict[str, Any]
) -> dict[str, Any]:
    """Sections that only appear on some deployments (GPU, CS internals, nonce)."""
    sections: dict[str, Any] = {}

    nvidia_cc = claims.get("nvidia_gpu", submods.get("nvidia_gpu", {}))
    if nvidia_cc:
        sections["gpu"] = nvidia_cc

    cs = {k: v for k, v in submods.items() if k.startswith("confidential_space")}
    if cs:
        sections["confidential_space"] = cs

    if claims.get("eat_nonce"):
        sections["eat_nonce"] = claims["eat_nonce"]
    return sections
