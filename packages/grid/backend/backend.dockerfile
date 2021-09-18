FROM python:3.9.6-slim as build

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends curl wget

WORKDIR /app
COPY ./requirements.txt /app

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN --mount=type=cache,target=/root/.cache \
  pip install --user "uvicorn[standard]" gunicorn
RUN --mount=type=cache,target=/root/.cache \
  pip install --user \
  torch==1.8.1+cpu torchvision==0.9.1+cpu torchcsprng==0.2.1+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html
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
COPY ./docker-scripts/start.sh /start.sh
COPY ./docker-scripts/gunicorn_conf.py /gunicorn_conf.py
COPY ./docker-scripts/start-reload.sh /start-reload.sh

RUN chmod +x /start.sh
RUN chmod +x /start-reload.sh

COPY --from=build /root/.local /root/.local
COPY --from=build /usr/local/bin/waitforit /usr/local/bin/waitforit

COPY ./ /app/
WORKDIR /app

# Celery worker
FROM backend as celery-worker
ENV C_FORCE_ROOT=1
RUN --mount=type=cache,target=/root/.cache \
  pip install --user watchdog pyyaml argh

COPY ./worker-start.sh /worker-start.sh
RUN chmod +x /worker-start.sh
CMD ["bash", "/worker-start.sh"]
