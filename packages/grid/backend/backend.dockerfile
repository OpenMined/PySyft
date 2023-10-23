ARG PYTHON_VERSION="3.11"
ARG TZ="Etc/UTC"
ARG SYFT_WORKDIR="/home/nonroot/app"
ARG NONROOT_UG="nonroot:nonroot"

# ==================== [BUILD STEP] Python Dev Base ==================== #

FROM cgr.dev/chainguard/wolfi-base as python_dev

ARG PYTHON_VERSION
ARG TZ

# Setup Python DEV
RUN apk update && \
    apk add build-base gcc tzdata python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ==================== [BUILD STEP] Install Syft Dependency ==================== #

FROM python_dev as syft_deps

ARG SYFT_WORKDIR
ARG NONROOT_UG

USER nonroot
WORKDIR $SYFT_WORKDIR
ENV PATH=$PATH:/home/nonroot/.local/bin

# copy skeleton to do package install
COPY --chown=$NONROOT_UG syft/setup.py ./syft/setup.py
COPY --chown=$NONROOT_UG syft/setup.cfg ./syft/setup.cfg
COPY --chown=$NONROOT_UG syft/pyproject.toml ./syft/pyproject.toml
COPY --chown=$NONROOT_UG syft/MANIFEST.in ./syft/MANIFEST.in
COPY --chown=$NONROOT_UG syft/src/syft/VERSION ./syft/src/syft/VERSION
COPY --chown=$NONROOT_UG syft/src/syft/capnp ./syft/src/syft/capnp

# Install all dependencies together here to avoid any version conflicts across pkgs
RUN --mount=type=cache,target=/home/nonroot/.cache/,rw,uid=65532 \
    pip install --user pip-autoremove jupyterlab==4.0.7 -e ./syft/ && \
    pip-autoremove ansible ansible-core -y

# ==================== [Final] Setup Syft Server ==================== #

FROM cgr.dev/chainguard/wolfi-base as backend

# inherit from global
ARG PYTHON_VERSION
ARG TZ
ARG SYFT_WORKDIR
ARG NONROOT_UG

# Setup Python
RUN apk update && \
    apk add --no-cache tzdata bash python-$PYTHON_VERSION py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/cache/apk/* && \
    mkdir -p /var/log/pygrid /home/nonroot/data/creds /home/nonroot/data/db && \
    chown -R $NONROOT_UG /var/log/pygrid /home/nonroot/data

USER nonroot
WORKDIR $SYFT_WORKDIR

# Update environment variables
ENV PATH=$PATH:/home/nonroot/.local/bin \
    PYTHONPATH=$SYFT_WORKDIR \
    APPDIR=$SYFT_WORKDIR \
    NODE_NAME="default_node_name" \
    NODE_TYPE="domain" \
    SERVICE_NAME="backend" \
    RELEASE="production" \
    DEV_MODE="False" \
    CONTAINER_HOST="docker" \
    PORT=80\
    HTTP_PORT=80 \
    HTTPS_PORT=443 \
    DOMAIN_CONNECTION_PORT=3030 \
    IGNORE_TLS_ERRORS="False" \
    DEFAULT_ROOT_EMAIL="info@openmined.org" \
    DEFAULT_ROOT_PASSWORD="changethis" \
    STACK_API_KEY="changeme" \
    MONGO_HOST="localhost" \
    MONGO_PORT="27017" \
    MONGO_USERNAME="root" \
    MONGO_PASSWORD="example" \
    CREDENTIALS_PATH="/home/nonroot/data/creds/credentials.json"

# Copy pre-built jupyterlab, syft dependencies
COPY --chown=$NONROOT_UG --from=syft_deps /home/nonroot/.local /home/nonroot/.local

# copy grid
COPY --chown=$NONROOT_UG grid/backend/grid ./grid

# copy syft
COPY --chown=$NONROOT_UG syft/ ./syft/

CMD ["bash", "./grid/start.sh"]
