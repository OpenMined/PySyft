# ==================== Base Image for with GPU (CUDA) support ==================== #
# Set arguments
ARG CUDA_VERSION="12.3.0"
ARG PYTHON_VERSION="3.11"
ARG TZ="Etc/UTC"
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
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04 as python_dev

# Set environment variables
ARG PYTHON_VERSION
ARG TZ
ARG USER
ARG UID

# Setup Python DEV
RUN apt update && apt upgrade -y && \
    apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev && \
    apt install -y libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev && \
    apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y && apt update &&\
    apt install -y python$PYTHON_VERSION python$PYTHON_VERSION-dev python$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# uncomment below for creating rootless user
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
RUN --mount=type=cache,target=$HOME/.cache/,rw,uid=0 \
    pip install --user torch==2.1.0+cu${CUDA_VERSION} -f https://download.pytorch.org/whl/cu${CUDA_VERSION}/torch_stable.html && \
    pip install --user pip-autoremove ./syft[data_science] && \
    pip-autoremove ansible ansible-core -y

# cache PIP_PACKAGES as a separate layer so installation will
# be faster when a new package is provided
RUN --mount=type=cache,target=$HOME/.cache/,rw \
    pip install --user $PIP_PACKAGES


# ==================== [Final] Setup Syft Server ==================== #
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04 as backend

# inherit from global
ARG APPDIR
ARG HOME
ARG PYTHON_VERSION
ARG TZ
ARG USER
ARG USER_GRP

# set up Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata bash python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip ${SYSTEM_PACKAGES} && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    apt-get clean && \
    # Uncomment for rootless user
    # useradd -m -u 1000 -s /bin/bash ${USER} && \
    mkdir -p /var/log/pygrid ${HOME}/data/creds ${HOME}/data/db ${HOME}/.cache ${HOME}/.local

# Setup final environment variables
# ENV NODE_NAME="default_node_name" \
#     NODE_TYPE="domain" \
#     SERVICE_NAME="backend" \
#     RELEASE="production" \
#     DEV_MODE="False" \
#     CONTAINER_HOST="docker" \
#     PORT=80 \
#     HTTP_PORT=80 \
#     HTTPS_PORT=443 \
#     DOMAIN_CONNECTION_PORT=3030 \
#     IGNORE_TLS_ERRORS="False" \
#     DEFAULT_ROOT_EMAIL="info@openmined.org" \
#     DEFAULT_ROOT_PASSWORD="changethis" \
#     STACK_API_KEY="changeme" \
#     MONGO_HOST="localhost" \
#     MONGO_PORT="27017" \
#     MONGO_USERNAME="root" \
#     MONGO_PASSWORD="example" \
#     CREDENTIALS_PATH="$HOME/data/creds/credentials.json"

# # Copy pre-built jupyterlab, syft dependencies
# COPY --chown=$USER_GRP --from=syft_deps $HOME/.local $HOME/.local

# # Copy grid
# COPY --chown=$USER_GRP grid/backend/grid ./grid

# # Copy syft
# COPY --chown=$USER_GRP syft/ ./syft/

# # Run any custom command
# RUN $CUSTOM_CMD

# # Start command
# CMD ["bash", "./grid/start.sh"]
