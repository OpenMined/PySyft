ARG PYTHON_VERSION='3.10.10'

FROM python:3.10.10-slim as build

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /root/.local

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl python3-dev gcc make build-essential cmake git

RUN --mount=type=cache,target=/root/.cache \
    pip install -U pip

# copy precompiled arm64 packages
COPY grid/backend/wheels /wheels

RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) != "x86_64" ]; then \
    pip install --user /wheels/jaxlib-0.3.14-cp310-none-manylinux2014_aarch64.whl; \
    pip install --user /wheels/tensorstore-0.1.25-cp310-cp310-linux_aarch64.whl; \
    fi

WORKDIR /app
COPY grid/worker/requirements.txt /app

RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

# Backend
FROM python:$PYTHON_VERSION-slim as worker
COPY --from=build /root/.local /root/.local

ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

RUN --mount=type=cache,target=/root/.cache \
    pip install -U pip

WORKDIR /app

# copy grid
COPY grid/worker /app/
COPY grid/backend/grid/bootstrap.py /app/bootstrap.py
RUN chmod +x /app/start.sh

# copy skeleton to do package install
COPY syft/setup.py /app/syft/setup.py
COPY syft/setup.cfg /app/syft/setup.cfg
COPY syft/pyproject.toml /app/syft/pyproject.toml
COPY syft/MANIFEST.in /app/syft/MANIFEST.in
COPY syft/src/syft/VERSION /app/syft/src/syft/VERSION
COPY syft/src/syft/capnp /app/syft/src/syft/capnp

# install syft
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -e /app/syft

# copy any changed source
COPY syft/src /app/syft/src

CMD ["bash", "/app/start.sh"]
