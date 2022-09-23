ARG PYTHON_VERSION='3.10.7'

FROM python:3.10.7-slim as build

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

RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) = "x86_64" ]; then \
  pip install --user torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
  fi

# copy precompiled arm64 packages
COPY grid/backend/wheels /wheels
# apple m1 build PyNaCl for aarch64
RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) != "x86_64" ]; then \
  # precompiled jaxlib and dm-tree
  pip install --user /wheels/jaxlib-0.3.14-cp310-none-manylinux2014_aarch64.whl; \
  tar -xvf /wheels/dm-tree-0.1.7.tar.gz; \
  pip install --user pytest-xdist[psutil]; \
  pip install --user torch==1.11.0 -f https://download.pytorch.org/whl/torch_stable.html; \
  git clone https://github.com/pybind/pybind11 && cd pybind11 && git checkout v2.6.2; \
  pip install --user dm-tree==0.1.7; \
  # fixes apple silicon in dev mode due to dependency from safety
  pip install --user ruamel.yaml==0.17.21; \
  fi

# install custom built python 3.10 wheel
RUN --mount=type=cache,target=/root/.cache pip install --user /wheels/tensorflow_federated-0.36.0-py2.py3-none-any.whl;

WORKDIR /app
COPY grid/backend/requirements.txt /app

RUN --mount=type=cache,target=/root/.cache \
  pip install --user -r requirements.txt

# Backend
FROM python:$PYTHON_VERSION-slim as backend
COPY --from=build /root/.local /root/.local

ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# copy start scripts and gunicorn conf
COPY grid/backend/docker-scripts/start.sh /start.sh
# COPY grid/backend/docker-scripts/gunicorn_conf.py /gunicorn_conf.py
COPY grid/backend/docker-scripts/start-reload.sh /start-reload.sh
COPY grid/backend/worker-start.sh /worker-start.sh
COPY grid/backend/worker-start-reload.sh /worker-start-reload.sh

RUN chmod +x /start.sh
RUN chmod +x /start-reload.sh
RUN chmod +x /worker-start.sh
RUN chmod +x /worker-start-reload.sh

RUN --mount=type=cache,target=/root/.cache \
  pip install -U pip

# allow container to wait for other services
RUN --mount=type=cache,target=/root/.cache \
  pip install --user watchdog pyyaml argh

WORKDIR /app

# copy grid
COPY grid/backend /app/

# copy syft
# until we have stable releases make sure to install syft
COPY syft/setup.py /app/syft/setup.py
COPY syft/setup.cfg /app/syft/setup.cfg
COPY syft/src /app/syft/src
RUN --mount=type=cache,target=/root/.cache \
  pip install --user -r requirements.txt

RUN pip install tensorflow-probability==0.18.0
RUN pip install tensorflow-federated==0.36.0

# install syft
RUN --mount=type=cache,target=/root/.cache \
  pip install --user -e /app/syft

# change to worker-start.sh or start-reload.sh as needed
CMD ["bash", "start.sh"]
