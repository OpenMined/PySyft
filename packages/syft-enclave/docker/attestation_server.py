"""
Syft Client Attestation Server

Serves the enclave's own attestation evidence on Tinfoil: the hardware
attestation document from the ``/tinfoil`` mount, plus the key bundle and
claims, fetched by peers over a connection pinned to the attested TLS key (see
``syft_enclaves.attestation.https``).

On Confidential Space ``/attestation`` is not served. The runner publishes the
launcher's signed token to peers through ``SYFT_version.json``; serving it here
as well would let a caller mint tokens carrying nonces of their choosing.

Nothing here is verified; a relying party appraises the evidence itself (see
``syft_enclaves.attestation``).
"""

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from syft_enclaves.attestation.envelope import AttestationKind
from syft_enclaves.evidence import probed_locations, select_provider
from syft_enclaves.evidence.key_bundle import (
    read_claims,
    read_public_bundle,
    sign_nonce,
)
from syft_enclaves.settings import AttestationSettings
from syft_enclaves.evidence.tee_token import validate_nonce

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
            "/attestation": "TEE attestation evidence (Tinfoil only)",
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
      nonce — optional freshness nonce, answered with a signature by the
              enclave's own key. It never goes into the hardware report.

    Not served on Confidential Space: peers read the token from
    ``SYFT_version.json``, and minting tokens here would let a caller choose
    what the TEE signs. Outside a TEE, returns deployment instructions instead.
    """
    if nonce is not None:
        error = validate_nonce(nonce)
        if error:
            return JSONResponse(status_code=400, content={"error": error})

    version = _get_syft_version()
    provider = _detect_provider()
    if provider is None:
        return _not_in_a_tee(version)
    if provider.kind == AttestationKind.CONFIDENTIAL_SPACE:
        return _not_served_on_confidential_space()

    try:
        evidence = provider.collect()
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
            # The runtime facts this enclave asserts about itself — email,
            # data owners, key bundle. Untrusted on their own; the signature
            # below is what makes them trustworthy on Tinfoil, and on
            # Confidential Space the token's nonce commits to them instead.
            "claims": read_claims(),
            # One signature over the caller's nonce AND the claims, by the
            # bundle's identity key: proof the enclave holds that key, that
            # this answer is for this exchange, and that these are its facts.
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


def _not_served_on_confidential_space() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": (
                "/attestation is not served on Confidential Space. The "
                "attestation token is published to SYFT_version.json."
            )
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
                "— see docs/terraform_cs.md."
            ),
            "tinfoil": (
                "Deploy a Tinfoil container from the config repo — see docs/tinfoil_deployment.md."
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
