ARG PYTHON_VERSION="3.11"
ARG TZ="Etc/UTC"
ARG SYFT_WORKDIR="/home/nonroot/app"

# ==================== [BUILD STEP] Build Base + rootless user ==================== #

FROM cgr.dev/chainguard/wolfi-base as python_dev

ARG PYTHON_VERSION
ARG TZ

# Setup Python DEV
RUN apk update && \
    apk add build-base gcc tzdata python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# # ==================== [BUILD STEP] Build+Install Jupyterlab ==================== #

FROM python_dev as jupyter

USER nonroot
WORKDIR /home/nonroot
ENV PATH=$PATH:/home/nonroot/.local/bin

RUN --mount=type=cache,target=/home/nonroot/.cache/,rw,uid=65532 \
    pip install --user jupyterlab==4.0.7

# ==================== [BUILD STEP] Build+Install ML Dependency ==================== #

FROM python_dev as syft_deps

ARG SYFT_WORKDIR

USER nonroot
WORKDIR $SYFT_WORKDIR
ENV PATH=$PATH:/home/nonroot/.local/bin

# copy skeleton to do package install
COPY --chown=nonroot:nonroot syft/setup.py ./syft/setup.py
COPY --chown=nonroot:nonroot syft/setup.cfg ./syft/setup.cfg
COPY --chown=nonroot:nonroot syft/pyproject.toml ./syft/pyproject.toml
COPY --chown=nonroot:nonroot syft/MANIFEST.in ./syft/MANIFEST.in
COPY --chown=nonroot:nonroot syft/src/syft/VERSION ./syft/src/syft/VERSION
COPY --chown=nonroot:nonroot syft/src/syft/capnp ./syft/src/syft/capnp

# Install Syft & dependencies
RUN --mount=type=cache,target=/home/nonroot/.cache/,rw,uid=65532 \
    pip install --user pip-autoremove && \
    pip install --user -e ./syft/ && \
    pip-autoremove ansible ansible-core -y

# ==================== [MAIN] Setup Syft  ==================== #

FROM cgr.dev/chainguard/wolfi-base as python_prod

# inherit from global
ARG PYTHON_VERSION
ARG TZ
ARG SYFT_WORKDIR

# Setup Python
RUN apk update && \
    apk add --no-cache tzdata bash python-$PYTHON_VERSION py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/cache/apk/*

USER nonroot
WORKDIR $SYFT_WORKDIR

# Update environment variables
ENV PATH=$PATH:/home/nonroot/.local/bin \
    PYTHONPATH=$SYFT_WORKDIR \
    APPDIR=$SYFT_WORKDIR

# Copy pre-built jupyterlab, syft dependencies
COPY --chown=nonroot:nonroot --from=syft_deps /home/nonroot/.local /home/nonroot/.local
COPY --chown=nonroot:nonroot --from=jupyter /home/nonroot/.local /home/nonroot/.local

# copy grid
COPY --chown=nonroot:nonroot grid/backend/grid ./grid

# copy syft
COPY --chown=nonroot:nonroot syft/ ./syft/

CMD ["bash", "./grid/start.sh"]
