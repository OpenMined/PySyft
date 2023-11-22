# ==================== Base Image for CPU workload ==================== #
ARG PYTHON_VERSION="3.11"
ARG TZ="Etc/UTC"

# change to USER="syftuser", UID=1000 and HOME="/home/$USER" for rootless
ARG USER="root"
ARG UID=0
ARG USER_GRP=$USER:$USER
ARG HOME="/root"
ARG APPDIR="$HOME/app"
# the DS provides the following args in his cog config file
ARG SYSTEM_PACKAGES=""
ARG PIP_PACKAGES="nothing --dry-run"
ARG CUSTOM_CMD="echo no custom commands passed"

# ==================== [BUILD STEP] Python Dev Base ==================== #

FROM cgr.dev/chainguard/wolfi-base as python_dev

ARG PYTHON_VERSION
ARG TZ
ARG USER
ARG UID

# Setup Python DEV
RUN --mount=type=cache,target=/var/cache/apk \
    apk update && \
    apk add build-base gcc tzdata python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# uncomment for creating rootless user
# && adduser -D -u $UID $USER

# ==================== [BUILD STEP] Install Syft Dependency ==================== #

FROM python_dev as syft_deps

ARG APPDIR
ARG HOME
ARG UID
ARG USER
ARG USER_GRP
ARG PIP_PACKAGES

USER $USER
WORKDIR $APPDIR
ENV PATH=$PATH:$HOME/.local/bin

# copy skeleton to do package install
COPY --chown=$USER_GRP syft/setup.py ./syft/setup.py
COPY --chown=$USER_GRP syft/setup.cfg ./syft/setup.cfg
COPY --chown=$USER_GRP syft/pyproject.toml ./syft/pyproject.toml
COPY --chown=$USER_GRP syft/MANIFEST.in ./syft/MANIFEST.in
COPY --chown=$USER_GRP syft/src/syft/VERSION ./syft/src/syft/VERSION
COPY --chown=$USER_GRP syft/src/syft/capnp ./syft/src/syft/capnp

# Install all dependencies together here to avoid any version conflicts across pkgs
RUN --mount=type=cache,target=$HOME/.cache/,rw,uid=$UID \
    pip install --user pip-autoremove ./syft && \
    pip-autoremove ansible ansible-core -y

# cache PIP_PACKAGES as a separate layer so installation will be faster when a new
# package is provided
RUN --mount=type=cache,target=$HOME/.cache/,rw \
    pip install --user $PIP_PACKAGES

# ==================== [Final] Setup Syft Server ==================== #

FROM cgr.dev/chainguard/wolfi-base as backend

# inherit from global
ARG APPDIR
ARG HOME
ARG PYTHON_VERSION
ARG TZ
ARG USER
ARG USER_GRP

# Setup Python
RUN --mount=type=cache,target=/var/cache/apk \
    apk update && \
    apk add tzdata bash python-$PYTHON_VERSION py$PYTHON_VERSION-pip $SYSTEM_PACKAGES && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    # Uncomment for rootless user
    # adduser -D -u 1000 $USER && \
    mkdir -p /var/log/pygrid $HOME/data/creds $HOME/data/db $HOME/.cache $HOME/.local
# chown -R $USER_GRP /var/log/pygrid $HOME/

USER $USER
WORKDIR $APPDIR

# Update environment variables
ENV PATH=$PATH:$HOME/.local/bin \
    PYTHONPATH=$APPDIR \
    APPDIR=$APPDIR \
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
    CREDENTIALS_PATH="$HOME/data/creds/credentials.json" \
    SYFT_BASE_IMAGE="True"

# Copy pre-built jupyterlab, syft dependencies
COPY --chown=$USER_GRP --from=syft_deps $HOME/.local $HOME/.local

# copy grid
COPY --chown=$USER_GRP grid/backend/grid ./grid

# copy syft
COPY --chown=$USER_GRP syft/ ./syft/

RUN $CUSTOM_CMD

CMD ["bash", "./grid/start.sh"]
