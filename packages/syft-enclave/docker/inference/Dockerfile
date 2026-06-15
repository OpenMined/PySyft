# Inference flavor of the enclave image: base enclave + pre-packaged Gemma
# inference pipeline behind /infer on the same port (8080).
# Build context is the monorepo root (set by packages/syft-enclave/Justfile).
ARG BASE_IMAGE=docker.io/openminedreleasebot/syft-client-enclave:latest
FROM ${BASE_IMAGE}

# Model deps (gemma → jax/flax/orbax) into the base image's venv.
COPY packages/syft-enclave/docker/inference/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /repo/.venv/bin/python -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /app
COPY packages/syft-enclave/docker/inference/inference_server.py .
COPY packages/syft-enclave/docker/inference/entrypoint.sh entrypoint-inference.sh
RUN chmod +x entrypoint-inference.sh

# Whitelist env vars that `tee-env-*` instance metadata may override.
# NOTE: this label REPLACES the base image's — it must contain the FULL list
# from docker/Dockerfile plus the inference vars. Keep both in sync.
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
SYFT_ENCLAVE_MODEL_OWNER,\
SYFT_ENCLAVE_MODEL_DATASET,\
SYFT_ENCLAVE_MODEL_SIZE,\
SYFT_ENCLAVE_LOGS_DATASET"

# EXPOSE 8080 and the /health HEALTHCHECK are inherited from the base image.
ENTRYPOINT ["./entrypoint-inference.sh"]
