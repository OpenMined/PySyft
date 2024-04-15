ARG PYTHON_VERSION="3.12"

# ==================== [BUILD STEP] Build Syft ==================== #

FROM cgr.dev/chainguard/wolfi-base as syft_deps

ARG PYTHON_VERSION

ENV PATH="/root/.local/bin:$PATH"

RUN apk update && apk upgrade && \
    apk add --no-cache build-base gcc python-$PYTHON_VERSION-dev-default py$PYTHON_VERSION-pip

COPY ./syft /tmp/syft

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    pip install --user jupyterlab==4.1.6 pip-autoremove==0.10.0 /tmp/syft && \
    pip-autoremove ansible ansible-core -y

# ==================== [Final] Setup Syft Client ==================== #

FROM cgr.dev/chainguard/wolfi-base as client

ARG PYTHON_VERSION

ENV PATH="/root/.local/bin:$PATH"

RUN apk update && apk upgrade && \
    apk add --no-cache git python-$PYTHON_VERSION-dev-default py$PYTHON_VERSION-pip

COPY --from=syft_deps /root/.local /root/.local

WORKDIR /root/notebooks/

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
