# Syft Worker
# Build as-is to create a base worker image
# Build with args to create a custom worker image

# NOTE: This dockerfile will be built inside a grid-backend container in PROD
# Hence COPY will not work the same way in DEV vs. PROD

# FIXME: Due to dependency on grid-backend base, python can only be changed from 3.11 to 3.11-dev
# Later we'd want to uninstall old python, and then install a new python runtime...
# ... but pre-built syft deps may break!

FROM openmined/grid-backend:0.8.4-beta.1

ARG PYTHON_VERSION="3.11"
ARG SYSTEM_PACKAGES=""
ARG PIP_PACKAGES="pip --dry-run"
ARG CUSTOM_CMD='echo "No custom commands passed"'

# Worker specific environment variables go here
ENV SYFT_WORKER="true"

RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    --mount=type=cache,target=$HOME/.cache/pip,sharing=locked \
    apk update && \
    apk add ${SYSTEM_PACKAGES} && \
    pip install --user ${PIP_PACKAGES} && \
    bash -c "$CUSTOM_CMD"
