# ==================== [BUILD STEP] Python Dev Base ==================== #

FROM cgr.dev/chainguard/wolfi-base as python_dev

ARG PYTHON_VERSION

# Setup Python DEV
RUN apk update && \
    apk add build-base gcc tzdata python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ==================== [BUILD STEP] Install Syft Dependency ==================== #
FROM python_dev as syft_deps

ARG USER_GRP

# copy skeleton to do package install
COPY --chown=$USER_GRP syft/setup.py ./syft/setup.py
# TODO: Separate out the DL/ML dependencies out of setup.cfg
COPY --chown=$USER_GRP syft/setup.cfg ./syft/setup.cfg
COPY --chown=$USER_GRP syft/pyproject.toml ./syft/pyproject.toml
COPY --chown=$USER_GRP syft/MANIFEST.in ./syft/MANIFEST.in
COPY --chown=$USER_GRP syft/src/syft/VERSION ./syft/src/syft/VERSION
COPY --chown=$USER_GRP syft/src/syft/capnp ./syft/src/syft/capnp
