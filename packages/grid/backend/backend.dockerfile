FROM python:3.10.4-slim as build

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
  DEBIAN_FRONTEND=noninteractive \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  curl python3-dev gcc make build-essential cmake

ENV WAITFORIT_VERSION="v2.4.1"
RUN curl -o /usr/local/bin/waitforit -sSL https://github.com/maxcnunes/waitforit/releases/download/$WAITFORIT_VERSION/waitforit-linux_amd64 && \
  chmod +x /usr/local/bin/waitforit

RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) = "x86_64" ]; then \
  pip install --user torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
  fi

# apple m1 build PyNaCl for aarch64
RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) != "x86_64" ]; then \
  pip install --user pytest-xdist[psutil]; \
  pip install --user torch==1.11.0 -f https://download.pytorch.org/whl/torch_stable.html; \
  fi

RUN --mount=type=cache,target=/root/.cache \
  pip install --user pycapnp==1.1.0; \
  pip install --user numpy==1.22.3; \
  pip install --user primesieve==2.3.0 --force-reinstall --no-cache-dir; \
  python -c "from primesieve.numpy._numpy import primes";

WORKDIR /app
COPY grid/backend/requirements.txt /app

RUN --mount=type=cache,target=/root/.cache \
  pip install --user -r requirements.txt

# Backend
FROM python:3.10.4-slim as backend
COPY --from=build /root/.local /root/.local
COPY --from=build /usr/local/bin/waitforit /usr/local/bin/waitforit

ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
  DEBIAN_FRONTEND=noninteractive \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  dnsutils iproute2

# copy start scripts and gunicorn conf
COPY grid/backend/docker-scripts/start.sh /start.sh
# COPY grid/backend/docker-scripts/gunicorn_conf.py /gunicorn_conf.py
COPY grid/backend/docker-scripts/start-reload.sh /start-reload.sh
COPY grid/backend/worker-start.sh /worker-start.sh
COPY grid/backend/worker-start-reload.sh /worker-start-reload.sh
COPY grid/backend/tailscale-gateway.sh /tailscale-gateway.sh

RUN chmod +x /start.sh
RUN chmod +x /start-reload.sh
RUN chmod +x /worker-start.sh
RUN chmod +x /worker-start-reload.sh
RUN chmod +x /tailscale-gateway.sh

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

# install syft
RUN --mount=type=cache,target=/root/.cache \
  pip install --user -e /app/syft

# change to worker-start.sh or start-reload.sh as needed
CMD ["bash", "start.sh"]
