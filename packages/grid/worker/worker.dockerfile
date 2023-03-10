ARG PYTHON_VERSION='3.10.10'

FROM python:3.10.10-slim as build

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl python3-dev gcc make build-essential cmake git

RUN --mount=type=cache,target=/root/.cache \
    pip install -U pip

# copy precompiled arm64 packages
COPY grid/backend/wheels /wheels
# apple m1 build PyNaCl for aarch64
RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) != "x86_64" ]; then \
    # precompiled jaxlib and dm-tree
    pip install --user /wheels/jaxlib-0.3.14-cp310-none-manylinux2014_aarch64.whl; \
    tar -xvf /wheels/dm-tree-0.1.7.tar.gz; \
    pip install --user pytest-xdist[psutil]; \
    git clone https://github.com/pybind/pybind11 && cd pybind11 && git checkout v2.6.2; \
    pip install --user dm-tree==0.1.7; \
    # fixes apple silicon in dev mode due to dependency from safety
    pip install --user ruamel.yaml==0.17.21; \
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
COPY grid/backend/grid/api/new/new_routes.py /app/new_routes.py
RUN chmod +x /app/start.sh

# copy skeleton to do package install
COPY syft/setup.py /app/syft/setup.py
COPY syft/setup.cfg /app/syft/setup.cfg
COPY syft/pyproject.toml /app/syft/pyproject.toml
COPY syft/MANIFEST.in /app/syft/MANIFEST.in
COPY syft/src/syft/VERSION /app/syft/src/syft/VERSION
COPY syft/src/syft/capnp /app/syft/src/syft/capnp
COPY syft/src/syft/cache /app/syft/src/syft/cache

# install syft
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -e /app/syft

# copy any changed source
COPY syft/src /app/syft/src

CMD ["bash", "/app/start.sh"]
