FROM cgr.dev/chainguard/wolfi-base as backend

ARG PYTHON_VERSION="3.12"

RUN apk update && apk upgrade && \
    apk add git bash python-$PYTHON_VERSION-default uv=0.1.22-r0

WORKDIR /root/app

# keep static deps separate to have each layer cached independently

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    uv venv && \
    uv pip install torch==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    uv pip install jupyterlab==4.1.5

COPY --chown=nonroot:nonroot \
    syft/setup.py syft/setup.cfg syft/pyproject.toml ./syft/

COPY --chown=nonroot:nonroot \
    syft/src/syft/VERSION ./syft/src/syft/

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    uv pip install  -e ./syft[data_science,telemetry] && \
    uv pip freeze | grep ansible | xargs uv pip uninstall

# Copy syft source (in rootless mode)

COPY --chown=nonroot:nonroot \
    grid/backend/grid grid/backend/worker_cpu.dockerfile ./grid/

# copy syft
COPY --chown=nonroot:nonroot \
    syft ./syft/

# Update environment variables
ENV \
    APPDIR="/root/app" \
    NODE_NAME="default_node_name" \
    NODE_TYPE="domain" \
    SERVICE_NAME="backend" \
    RELEASE="production" \
    DEV_MODE="False" \
    DEBUGGER_ENABLED="False" \
    CONTAINER_HOST="docker" \
    OBLV_ENABLED="False" \
    OBLV_LOCALHOST_PORT=3030 \
    DEFAULT_ROOT_EMAIL="info@openmined.org" \
    DEFAULT_ROOT_PASSWORD="changethis" \
    STACK_API_KEY="changeme" \
    MONGO_HOST="localhost" \
    MONGO_PORT="27017" \
    MONGO_USERNAME="root" \
    MONGO_PASSWORD="example"

CMD ["bash", "./grid/start.sh"]
