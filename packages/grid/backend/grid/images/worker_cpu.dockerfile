# Syft Worker
# Build as-is to create a base worker image
# Build with args to create a custom worker image

# NOTE: This dockerfile will be built inside a syft-backend container in PROD
# Hence COPY will not work the same way in DEV vs. PROD

ARG SYFT_VERSION_TAG="0.9.2-beta.3"
FROM openmined/syft-backend:${SYFT_VERSION_TAG}

# should match base image python version
ARG PYTHON_VERSION="3.12"
ARG SYSTEM_PACKAGES=""
ARG PIP_PACKAGES="pip --dry-run"
ARG CUSTOM_CMD='echo "No custom commands passed"'

# Worker specific environment variables go here
ENV SYFT_WORKER="true" \
    SYFT_VERSION_TAG=${SYFT_VERSION_TAG} \
    UV_HTTP_TIMEOUT=600

# dont run `apk upgrade` here, as it runs upgrades on the base image
# which may break syft or carry over breaking changes by wolfi-os
RUN apk update && \
    apk add --no-cache ${SYSTEM_PACKAGES} && \
    # if uv is present then run uv pip install else simple pip install
    if [ -x "$(command -v uv)" ]; then uv pip install --no-cache ${PIP_PACKAGES}; else pip install --user ${PIP_PACKAGES}; fi && \
    bash -c "$CUSTOM_CMD"
