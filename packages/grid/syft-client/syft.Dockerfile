ARG PYTHON_VERSION="3.12"
ARG TORCH_VERSION="2.3.1"

# ==================== [BUILD STEP] Build Syft ==================== #

FROM cgr.dev/chainguard/wolfi-base as syft_deps

ARG PYTHON_VERSION
ARG TORCH_VERSION

ENV PATH="/root/.local/bin:$PATH"

# Setup Python DEV
RUN apk update && apk upgrade && \
    apk add build-base gcc python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip && \
    # preemptive fix for wolfi-os breaking python entrypoint
    (test -f /usr/bin/python || ln -s /usr/bin/python3.12 /usr/bin/python)

# keep static deps separate to have each layer cached independently
# if amd64 then we need to append +cpu to the torch version
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    ARCH=$(arch | sed s/aarch64/arm64/ | sed s/x86_64/amd64/) && \
    if [[ "$ARCH" = "amd64" ]]; then TORCH_VERSION="$TORCH_VERSION+cpu"; fi && \
    pip install --user torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu

COPY ./syft /tmp/syft

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    # remove torch because we already have the cpu version pre-installed
    sed --in-place /torch==/d ./tmp/syft/setup.cfg && \
    pip install --user jupyterlab==4.2.2 ./tmp/syft[data_science]

# ==================== [Final] Setup Syft Client ==================== #

FROM cgr.dev/chainguard/wolfi-base as client

ARG PYTHON_VERSION

ENV PATH="/root/.local/bin:$PATH"

RUN apk update && apk upgrade && \
    apk add --no-cache git python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip

COPY --from=syft_deps /root/.local /root/.local

WORKDIR /root/notebooks/

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
