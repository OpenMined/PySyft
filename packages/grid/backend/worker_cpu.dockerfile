# Syft Worker
# Build as-is to create a base worker image
# Build with args to create a custom worker image

# NOTE: This dockerfile will be built inside a grid-backend container in PROD
# Hence COPY will not work the same way in DEV vs. PROD

# FIXME: Due to dependency on grid-backend base, python can only be changed from 3.11 to 3.11-dev
# Later we'd want to uninstall old python, and then install a new python runtime...
# ... but pre-built syft deps may break!

ARG SYFT_VERSION_TAG="0.8.7-beta.2"
FROM openmined/grid-backend:${SYFT_VERSION_TAG}

ARG PYTHON_VERSION="3.12"
ARG SYSTEM_PACKAGES=""
ARG PIP_PACKAGES="pip --dry-run"
ARG CUSTOM_CMD='echo "No custom commands passed"'

# Worker specific environment variables go here
ENV SYFT_WORKER="true" \
    SYFT_VERSION_TAG=${SYFT_VERSION_TAG} \
    UV_HTTP_TIMEOUT=600

RUN apk update && apk upgrade && \
    apk add --no-cache ${SYSTEM_PACKAGES} && \
    # if uv is present then run uv pip install else simple pip install
    if [ -x "$(command -v uv)" ]; then uv pip install --no-cache ${PIP_PACKAGES}; else pip install --user ${PIP_PACKAGES}; fi && \
    bash -c "$CUSTOM_CMD"
