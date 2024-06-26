ARG PYTHON_VERSION="3.12"

# ==================== [BUILD STEP] Build Syft ==================== #

FROM cgr.dev/chainguard/wolfi-base as syft_deps

ARG PYTHON_VERSION

ENV PATH="/root/.local/bin:$PATH"

# Setup Python DEV
RUN apk update && apk upgrade && \
    apk add build-base gcc python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip && \
    # preemptive fix for wolfi-os breaking python entrypoint
    (test -f /usr/bin/python || ln -s /usr/bin/python3.12 /usr/bin/python)

COPY ./syft /tmp/syft

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    pip install --user jupyterlab==4.2.2 /tmp/syft

# ==================== [Final] Setup Syft Client ==================== #

FROM cgr.dev/chainguard/wolfi-base as client

ARG PYTHON_VERSION

ENV PATH="/root/.local/bin:$PATH"

RUN apk update && apk upgrade && \
    apk add --no-cache git python-$PYTHON_VERSION-dev py$PYTHON_VERSION-pip

COPY --from=syft_deps /root/.local /root/.local

WORKDIR /root/notebooks/

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
