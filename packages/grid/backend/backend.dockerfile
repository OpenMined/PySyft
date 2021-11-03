FROM python:3.9.6-slim as build

RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends curl wget

# apple m1
RUN if [ $(uname -m) != "x86_64" ]; then \
  apt-get update && apt-get install -y --no-install-recommends libsodium-dev; \
  fi

WORKDIR /app
COPY grid/backend/requirements.txt /app

# apple m1
RUN if [ $(uname -m) != "x86_64" ]; then \
  pip install --user uvicorn gunicorn; \
  fi

# Allow installing dev dependencies to run tests
RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) = "x86_64" ]; then \
  pip install --user "uvicorn[standard]" gunicorn; \
  fi

# apple m1
RUN if [ $(uname -m) != "x86_64" ]; then \
  pip install --user torch==1.8.1; \
  fi

RUN --mount=type=cache,target=/root/.cache \
  if [ $(uname -m) = "x86_64" ]; then pip install --user \
  torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
  fi

RUN --mount=type=cache,target=/root/.cache \
  pip install --user -r requirements.txt

# allow container to wait for other services
ENV WAITFORIT_VERSION="v2.4.1"
RUN curl -o /usr/local/bin/waitforit -sSL https://github.com/maxcnunes/waitforit/releases/download/$WAITFORIT_VERSION/waitforit-linux_amd64 && \
  chmod +x /usr/local/bin/waitforit

# Backend
FROM python:3.9.6-slim as backend
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# copy start scripts and gunicorn conf
COPY grid/backend/docker-scripts/start.sh /start.sh
COPY grid/backend/docker-scripts/gunicorn_conf.py /gunicorn_conf.py
COPY grid/backend/docker-scripts/start-reload.sh /start-reload.sh
COPY grid/backend/worker-start.sh /worker-start.sh

RUN chmod +x /start.sh
RUN chmod +x /start-reload.sh
RUN chmod +x /worker-start.sh

COPY --from=build /root/.local /root/.local
COPY --from=build /usr/local/bin/waitforit /usr/local/bin/waitforit

RUN --mount=type=cache,target=/root/.cache \
  pip install --user watchdog pyyaml argh

WORKDIR /app

# apple m1
# Download PyNacl for arm64
RUN if [ $(uname -m) != "x86_64" ]; then \
  wget https://opencomputinglab.github.io/vce-wheelhouse/wheelhouse/PyNaCl-1.4.0-cp39-cp39-linux_aarch64.whl && \
  pip install --user ./PyNaCl-1.4.0-cp39-cp39-linux_aarch64.whl && \
  rm ./PyNaCl-1.4.0-cp39-cp39-linux_aarch64.whl; \
  fi

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
