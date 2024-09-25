ARG PYTHON_VERSION="3.12"
ARG UV_VERSION="0.2.13-r0"
ARG TORCH_VERSION="2.2.2"

# wolfi-os pkg definition links
# https://github.com/wolfi-dev/os/blob/main/python-3.12.yaml
# https://github.com/wolfi-dev/os/blob/main/py3-pip.yaml
# https://github.com/wolfi-dev/os/blob/main/uv.yaml

# ==================== [BUILD STEP] Python Dev Base ==================== #

FROM cgr.dev/chainguard/wolfi-base AS syft_deps

ARG PYTHON_VERSION
ARG UV_VERSION
ARG TORCH_VERSION

# Setup Python DEV
RUN apk update && apk upgrade && \
    apk add build-base gcc python-$PYTHON_VERSION-dev uv=$UV_VERSION && \
    # preemptive fix for wolfi-os breaking python entrypoint
    (test -f /usr/bin/python || ln -s /usr/bin/python3.12 /usr/bin/python)

WORKDIR /root/app

ENV UV_HTTP_TIMEOUT=600

# keep static deps separate to have each layer cached independently
# if amd64 then we need to append +cpu to the torch version
# uv issues: https://github.com/astral-sh/uv/issues/3437 & https://github.com/astral-sh/uv/issues/2541
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    uv venv && \
    ARCH=$(arch | sed s/aarch64/arm64/ | sed s/x86_64/amd64/) && \
    if [[ "$ARCH" = "amd64" ]]; then TORCH_VERSION="$TORCH_VERSION+cpu"; fi && \
    uv pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu

COPY syft/setup.py syft/setup.cfg syft/pyproject.toml ./syft/

COPY syft/src/syft/VERSION ./syft/src/syft/

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    # remove torch because we already have the cpu version pre-installed
    sed --in-place /torch==/d ./syft/setup.cfg && \
    uv pip install -e ./syft[data_science,telemetry]

# ==================== [Final] Setup Syft Server ==================== #

FROM cgr.dev/chainguard/wolfi-base AS backend

ARG PYTHON_VERSION
ARG UV_VERSION

RUN apk update && apk upgrade && \
    apk add --no-cache git bash python-$PYTHON_VERSION py$PYTHON_VERSION-pip uv=$UV_VERSION && \
    # preemptive fix for wolfi-os breaking python entrypoint
    (test -f /usr/bin/python || ln -s /usr/bin/python3.12 /usr/bin/python)

WORKDIR /root/app/

# Copy pre-built syft dependencies
COPY --from=syft_deps /root/app/.venv .venv

# copy server
COPY grid/backend/grid ./grid/

# copy syft
COPY syft ./syft/

# Update environment variables
ENV \
    # "activate" venv
    PATH="/root/app/.venv/bin/:$PATH" \
    VIRTUAL_ENV="/root/app/.venv" \
    # Syft
    APPDIR="/root/app" \
    SERVER_NAME="default_server_name" \
    SERVER_TYPE="datasite" \
    SERVER_SIDE_TYPE="high" \
    RELEASE="production" \
    DEV_MODE="False" \
    DEBUGGER_ENABLED="False" \
    TRACING="False" \
    CONTAINER_HOST="docker" \
    DEFAULT_ROOT_EMAIL="info@openmined.org" \
    DEFAULT_ROOT_PASSWORD="changethis" \
    STACK_API_KEY="changeme" \
    POSTGRESQL_DBNAME="syftdb_postgres" \
    POSTGRESQL_HOST="localhost" \
    POSTGRESQL_PORT="5432" \
    POSTGRESQL_USERNAME="syft_postgres" \
    POSTGRESQL_PASSWORD="example"

CMD ["bash", "./grid/start.sh"]
