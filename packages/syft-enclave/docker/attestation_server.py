"""
Syft Client Attestation Server

When running inside Google Confidential Spaces, this server fetches
the TEE attestation token from the Confidential Space launcher and
displays it as structured JSON.

The attestation token is a signed JWT issued by Google's Confidential
Computing attestation service. It contains cryptographic proof of:
  - The hardware TEE type (AMD SEV-SNP, Intel TDX)
  - Secure boot status
  - The exact container image digest running
  - Debug status of the VM
  - GPU confidential computing mode (if applicable)

Architecture:
  Container -> Unix socket (/run/container_launcher/teeserver.sock)
            -> Confidential Space Launcher
            -> Google Attestation Service
            -> Signed JWT returned to container
"""

import base64
import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from syft_enclaves.tee_token import (
    TEE_SOCKET_PATH,
    build_eat_nonce,
    fetch_attestation_token,
    validate_nonce,
)

app = FastAPI(title="Syft Client Enclave", version="0.1.0")


def _get_syft_version() -> str:
    try:
        from syft_client import __version__

        return __version__
    except ImportError:
        return "unknown"


def _is_confidential_space() -> bool:
    """Check if we're running inside Google Confidential Spaces."""
    return os.path.exists(str(TEE_SOCKET_PATH))


def _decode_jwt_payload(token: str) -> dict:
    """Base64-decode the JWT payload (middle segment) without signature verification.

    We decode without verification because the purpose of this endpoint is to
    DISPLAY the attestation claims. The token itself (returned in raw_token)
    is what a relying party would verify against Google's JWKS endpoint.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid JWT: expected 3 parts, got {len(parts)}")

    # JWT base64url encoding — add padding if needed
    payload_b64 = parts[1]
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding

    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    return json.loads(payload_bytes)


def _format_timestamp(epoch: int | float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _structure_claims(claims: dict) -> dict:
    """Organize raw JWT claims into logical sections for display."""
    submods = claims.get("submods", {})
    container = submods.get("container", {})
    gce = submods.get("gce", {})
    cs = {}
    for key, val in submods.items():
        if key.startswith("confidential_space"):
            cs[key] = val

    result = {
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

    # GPU confidential computing claims (if present)
    nvidia_cc = claims.get("nvidia_gpu", submods.get("nvidia_gpu", {}))
    if nvidia_cc:
        result["gpu"] = nvidia_cc

    # Confidential Space-specific claims
    if cs:
        result["confidential_space"] = cs

    eat_nonce_raw = claims.get("eat_nonce")
    if eat_nonce_raw:
        result["eat_nonce"] = eat_nonce_raw

    return result


@app.get("/")
def index():
    """Landing page with syft-client info and available endpoints."""
    return {
        "service": "syft-client-enclave",
        "syft_client_version": _get_syft_version(),
        "confidential_space_detected": _is_confidential_space(),
        "endpoints": {
            "/": "This page",
            "/attestation": "TEE attestation report (requires Confidential Spaces)",
            "/health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/attestation")
def attestation(nonce: str | None = None):
    """Fetch and display the TEE attestation report.

    Query params:
      nonce — optional caller-supplied freshness nonce. When provided it is
              embedded in the signed JWT via ``eat_nonce`` so the caller can
              verify the report was generated on demand (not replayed).

    When running in Confidential Spaces:
      1. Validates the nonce (if supplied)
      2. Builds an eat_nonce array (version hash + optional caller nonce)
      3. Requests an OIDC attestation token with eat_nonce
      4. Decodes the JWT claims
      5. Returns structured attestation data + the raw token

    When NOT in Confidential Spaces:
      Returns instructions for how to deploy correctly.
    """
    if nonce is not None:
        error = validate_nonce(nonce)
        if error:
            return JSONResponse(status_code=400, content={"error": error})

    version = _get_syft_version()

    if not _is_confidential_space():
        return {
            "status": "not_in_confidential_space",
            "syft_client_version": version,
            "message": (
                "Attestation unavailable. This container must run on "
                "Google Confidential Spaces. The TEE socket at "
                f"{TEE_SOCKET_PATH} was not found."
            ),
            "instructions": {
                "build": "docker build -t syft-client-enclave -f docker/Dockerfile .",
                "deploy": (
                    "Deploy on a Confidential VM with the Confidential Spaces "
                    "image to enable attestation."
                ),
            },
        }

    try:
        eat_nonce = build_eat_nonce(caller_nonce=nonce)
        raw_token = fetch_attestation_token(eat_nonce=eat_nonce)
        claims = _decode_jwt_payload(raw_token)
        structured = _structure_claims(claims)

        return {
            "status": "running_in_confidential_space",
            "syft_client_version": version,
            "attestation": structured,
            "nonce_info": {
                "version": eat_nonce[0],
                "caller_nonce": nonce,
            },
            "raw_token": raw_token,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "attestation_error",
                "syft_client_version": version,
                "error": str(e),
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
