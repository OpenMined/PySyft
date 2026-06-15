FROM python:3.12-slim

RUN pip install uv

# Build context is the monorepo root (set by packages/syft-enclave/Justfile).
# We install syft-enclave from the workspace so syft-client + its workspace
# peers (syft-bg, syft-job, syft-dataset, syft-permissions, syft-perms) resolve
# from packages/* instead of PyPI.
WORKDIR /repo
COPY pyproject.toml uv.lock README.md ./
COPY syft_client ./syft_client
COPY packages ./packages

# Sync just the deps of the syft-enclave package into /repo/.venv.
RUN uv sync --frozen --no-dev --package syft-enclave

# FastAPI + uvicorn for the attestation server (not part of syft-enclave's own
# deps; only needed inside the image).
COPY packages/syft-enclave/docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /repo/.venv/bin/python -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENV PATH="/repo/.venv/bin:$PATH"

WORKDIR /app
COPY packages/syft-enclave/docker/attestation_server.py .
COPY packages/syft-enclave/docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Whitelist env vars that `tee-env-*` instance metadata is allowed to override.
# Without this, the Confidential Spaces launcher rejects every override.
LABEL "tee.launch_policy.allow_env_override"="\
SYFT_ENCLAVE_EMAIL,\
SYFT_ENCLAVE_DATA_OWNERS,\
SYFT_ENCLAVE_REQUIRE_TEE,\
SYFT_ENCLAVE_FRESH_STATE,\
SYFT_ENCLAVE_USE_ENCRYPTION,\
SYFT_DEFAULT_JOB_TIMEOUT_SECONDS,\
SYFT_BOOTSTRAP,\
SYFT_BOOTSTRAP_WIF_AUDIENCE,\
SYFT_BOOTSTRAP_WIF_SECRET,\
SYFT_BOOTSTRAP_SA_SECRET,\
SYFT_ENCLAVE_TOKEN_CONTENT"

# Confidential Spaces expects port 8080
# Note: runs as root because the TEE attestation socket
# at /run/container_launcher/teeserver.sock is root-owned.
# In a TEE, security comes from hardware isolation, not container user permissions.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
