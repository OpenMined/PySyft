# ==================== [BASE] Setup deps + rootless user ==================== #

FROM cgr.dev/chainguard/wolfi-base as base

ARG PYTHON_VERSION="3.11"
ENV TZ=Etc/UTC

RUN --mount=type=cache,target=/var/cache/apk \
    apk update && \
    apk add bash tzdata python-$PYTHON_VERSION py$PYTHON_VERSION-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG HOME=/home/nonroot
ARG NONROOT=nonroot:nonroot

# use wolfi-base provided rootless user
USER nonroot

ENV PATH=$PATH:$HOME/.local/bin

# ==================== [BUILD CACHE] Jupyterlab Cache ==================== #

FROM base as jupyter

RUN --mount=type=cache,target=$HOME/.cache/ \
    pip install --user jupyterlab

# ==================== [BUILD CACHE] Syft deps changes ==================== #

FROM base as syft_deps_changes

WORKDIR $HOME/

COPY --chown=$NONROOT syft/setup.cfg $HOME/setup.cfg

# setup.cfg might change, but ML dependencies may not
# so we take a snapshot of everything in DOCKER:CACHED
RUN awk '/DOCKER:CACHED:START/,/DOCKER:CACHED:END/ {if (!/#/) print}' $HOME/setup.cfg | sed 's/ //g' | sort > $HOME/requirements.txt;

# ==================== [BUILD CACHE] Syft ML Cache ==================== #

FROM base as syft_cached_deps

COPY --from=syft_deps_changes $HOME/requirements.txt $HOME/requirements.txt

# Should hopefully be run only when req.txt changes
RUN --mount=type=cache,target=$HOME/.cache/ \
    pip install --user -r $HOME/requirements.txt && \
    rm $HOME/requirements.txt

# ==================== [MAIN] Setup Syft ==================== #

FROM syft_cached_deps

# Copy pre-built jupyterlab
COPY --from=jupyter --chown=$NONROOT $HOME/.local $HOME/.local

WORKDIR $HOME/app
ENV PYTHONPATH=$HOME/app
ENV APPDIR=$HOME/app

# copy skeleton to do package install
COPY --chown=$NONROOT syft/setup.py ./syft/setup.py
COPY --chown=$NONROOT syft/setup.cfg ./syft/setup.cfg
COPY --chown=$NONROOT syft/pyproject.toml ./syft/pyproject.toml
COPY --chown=$NONROOT syft/MANIFEST.in ./syft/MANIFEST.in
COPY --chown=$NONROOT syft/src/syft/VERSION ./syft/src/syft/VERSION
COPY --chown=$NONROOT syft/src/syft/capnp ./syft/src/syft/capnp

# Install Syft
RUN pip install --user pip-autoremove && \
    pip install --user -e ./syft/ && \
    pip-autoremove ansible ansible-core -y && \
    pip uninstall pip-autoremove -y && \
    rm -rf $HOME/.cache/

# copy grid
COPY --chown=$NONROOT grid/backend .

# copy any changed source
COPY --chown=$NONROOT syft/src ./syft/src

# change to worker-start.sh or start-reload.sh as needed
CMD ["bash", "./grid/start.sh"]
