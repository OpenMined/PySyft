"""
Syft Client Attestation Server

Operator-facing view of the enclave's own attestation evidence, whichever
deployment target it is running on. The evidence itself comes from
``syft_enclaves.evidence``:

  - Confidential Space: a signed JWT from the launcher socket, proving the
    hardware TEE type, secure boot, debug status and container image digest.
  - Tinfoil: the hardware attestation document from the ``/tinfoil`` mount.

This endpoint is for humans and for ``just attest`` / ``just tinfoil-attest``.
Production publishes evidence to peers through ``SYFT_version.json``; nothing
here is verified, and a relying party appraises the evidence itself (see
``syft_enclaves.attestation``).
"""

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from syft_enclaves.evidence import probed_locations, select_provider
from syft_enclaves.evidence.key_bundle import read_public_bundle, sign_nonce
from syft_enclaves.settings import AttestationSettings
from syft_enclaves.tee_token import validate_nonce

app = FastAPI(title="Syft Client Enclave", version="0.1.0")


def _get_syft_version() -> str:
    try:
        from syft import __version__

        return __version__
    except ImportError:
        return "unknown"


def _detect_provider():
    """The configured provider for this deployment target, or None outside a TEE.

    Reads settings per request rather than at import so the endpoint reflects
    the environment the container was actually started with.
    """
    settings = AttestationSettings()
    return select_provider(settings.attestation_provider, settings)


@app.get("/")
def index():
    """Landing page with syft info and available endpoints."""
    provider = _detect_provider()
    return {
        "service": "syft-enclave",
        "syft_version": _get_syft_version(),
        "attestation_provider": provider.kind.value if provider else None,
        "endpoints": {
            "/": "This page",
            "/attestation": "TEE attestation evidence (requires a TEE deployment)",
            "/health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/attestation")
def attestation(nonce: str | None = None):
    """Show this enclave's own attestation evidence.

    Query params:
      nonce — optional freshness nonce. Only Confidential Space can bind one
              into the evidence (via ``eat_nonce``); Tinfoil rejects it,
              because its report's user data is fully used by the shim's keys
              and the attestation document is a static file.

    Outside a TEE, returns deployment instructions instead.
    """
    if nonce is not None:
        error = validate_nonce(nonce)
        if error:
            return JSONResponse(status_code=400, content={"error": error})

    version = _get_syft_version()
    provider = _detect_provider()
    if provider is None:
        return _not_in_a_tee(version)

    try:
        # Only some TEEs can bind a caller nonce into the report itself.
        # Where they cannot, the nonce is still answered — signed with the
        # enclave's own key below — so the caller gets freshness either way.
        evidence = (
            provider.collect(caller_nonce=nonce)
            if nonce and provider.accepts_caller_nonce
            else provider.collect()
        )
        return {
            "status": "running_in_tee",
            "provider": provider.kind.value,
            "syft_version": version,
            "attestation": provider.describe(evidence),
            "evidence": evidence.to_version_field(),
            # The enclave's syft public keys. A caller that pinned this
            # connection to the TLS key the report commits to can trust these;
            # over an unpinned connection they are worth nothing. None when
            # encryption is disabled or the runner has not started yet.
            "key_bundle": read_public_bundle(),
            # Proof the enclave holds the private half of that bundle, and
            # that this response was produced for this exchange: a signature
            # over the caller's nonce by the bundle's identity key.
            "nonce": nonce,
            "nonce_signature": sign_nonce(nonce) if nonce else None,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "attestation_error",
                "provider": provider.kind.value,
                "syft_version": version,
                "error": str(e),
            },
        )


def _not_in_a_tee(version: str) -> dict:
    return {
        "status": "not_in_a_tee",
        "syft_version": version,
        "message": (
            f"Attestation unavailable: no TEE detected. Probed: {probed_locations()}."
        ),
        "instructions": {
            "build": "docker build -t syft-enclave -f docker/Dockerfile .",
            "confidential_space": (
                "Deploy on a Confidential VM with the Confidential Space image "
                "— see docs/terraform.md."
            ),
            "tinfoil": (
                "Deploy a Tinfoil container from the config repo — see docs/tinfoil.md."
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
