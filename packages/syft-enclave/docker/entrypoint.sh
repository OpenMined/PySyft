#!/bin/bash
set -euo pipefail

# Marker paths for the two supported deployment targets; kept in step with
# the providers in src/syft_enclaves/providers/.
CS_TEE_SOCKET="/run/container_launcher/teeserver.sock"
TINFOIL_ATTESTATION="/tinfoil/attestation.json"

echo "=== Syft Enclave Server ==="
echo "syft version: $(python -c 'from syft import __version__; print(__version__)' 2>/dev/null || echo 'unknown')"

# Token bootstrap. SYFT_BOOTSTRAP picks the provider; see
# src/syft_enclaves/bootstrap.py for details.
: "${SYFT_ENCLAVE_TOKEN_PATH:=/run/syft-enclave/token.json}"
export SYFT_ENCLAVE_TOKEN_PATH
python -m syft_enclaves.bootstrap

if [ -S "$CS_TEE_SOCKET" ]; then
    echo "Confidential Space detected: launcher socket at $CS_TEE_SOCKET"
elif [ -f "$TINFOIL_ATTESTATION" ]; then
    echo "Tinfoil detected: attestation document at $TINFOIL_ATTESTATION"
else
    echo "WARNING: no TEE detected (looked for $CS_TEE_SOCKET and $TINFOIL_ATTESTATION)"
    echo "The attestation endpoint will return deployment instructions instead of real evidence."
    echo "See docs/terraform.md (Confidential Spaces) or docs/tinfoil.md (Tinfoil)."
fi

# Attestation server (background) — configured via PORT.
PORT="${PORT:-8080}"
echo "Starting attestation server on port $PORT..."
uvicorn attestation_server:app --host 0.0.0.0 --port "$PORT" &

# Enclave runner (foreground) — configured entirely via SYFT_ENCLAVE_* env vars.
# See syft_enclaves.settings.EnclaveSettings for the full list of variables.
echo "Starting enclave runner..."
exec python -m syft_enclaves
