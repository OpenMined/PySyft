ARG PYTHON_VERSION='3.11'

FROM python:3.11-slim as build

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /root/.local

RUN apt-get update && apt-get upgrade -y
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
    pip install --user /wheels/jaxlib-0.4.10-cp311-cp311-manylinux2014_aarch64.whl; \
    fi

WORKDIR /app

# Backend
FROM python:$PYTHON_VERSION-slim as worker
RUN apt-get update && apt-get upgrade -y
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
    pip install --user -e /app/syft && \
    pip uninstall ansible ansible-core -y && \
    rm -rf ~/.local/lib/python3.11/site-packages/ansible_collections

# clean up
RUN apt purge --auto-remove linux-libc-dev -y

# copy any changed source
COPY syft/src /app/syft/src

CMD ["bash", "/app/start.sh"]
