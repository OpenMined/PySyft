FROM ubuntu:24.04

# Ubuntu 24.04 LTS ships Python 3.12 as its system interpreter, which is the
# version the workspace targets. python3-venv is separate on Debian/Ubuntu and
# uv needs it to build /repo/.venv; ca-certificates is required for TLS to the
# package index. uv is installed from its standalone distribution rather than
# pip, since Ubuntu marks the system interpreter externally managed (PEP 668).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

# Build context is the monorepo root (set by packages/syft-enclave/Justfile).
# We install syft-enclave from the workspace so syft + its workspace
# peers (syft-bg, syft-job, syft-dataset, syft-permissions, syft-perms) resolve
# from packages/* instead of PyPI.
WORKDIR /repo
COPY pyproject.toml uv.lock README.md ./
COPY syft ./syft
COPY packages ./packages

# Sync just the deps of the syft-enclave package into /repo/.venv.
RUN uv sync --frozen --no-dev --package syft-enclave

# FastAPI + uvicorn for the attestation server (not part of syft-enclave's own
# deps; only needed inside the image).
COPY packages/syft-enclave/docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /repo/.venv/bin/python -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENV PATH="/repo/.venv/bin:$PATH"

# GPU deployments: Confidential Space installs the NVIDIA driver at
# /usr/local/nvidia (tee-install-gpu-driver=true) and the workload must have
# its libs on LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64

# Unprivileged account that sandboxed jobs run as (SYFT_JOB_SANDBOX). Jobs are
# untrusted code; running them as root would let them read the Drive token and
# rewrite the enclave's own source. Nologin: it exists to own processes, not to
# be logged into.
RUN useradd --uid 1500 --no-create-home --shell /usr/sbin/nologin syftjob

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
SYFT_ENCLAVE_TOKEN_CONTENT,\
SYFT_JOB_SANDBOX,\
SYFT_JOB_SANDBOX_UID,\
SYFT_JOB_SANDBOX_GID"


LABEL "tee.launch_policy.log_redirect"="always"

# Confidential Spaces expects port 8080
# Note: runs as root because the TEE attestation socket
# at /run/container_launcher/teeserver.sock is root-owned.
# In a TEE, security comes from hardware isolation, not container user permissions.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
