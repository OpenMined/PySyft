FROM python:3.9.6-slim as backend

RUN \
  apt-get update && \
  apt-get upgrade -y && \
  apt-get clean && \
  apt-get install -y --no-install-recommends curl wget

# start scripts and gunicorn conf (from tiangolo)
COPY ./docker-scripts/start.sh /start.sh
COPY ./docker-scripts/gunicorn_conf.py /gunicorn_conf.py
COPY ./docker-scripts/start-reload.sh /start-reload.sh
RUN chmod +x /start.sh
RUN chmod +x /start-reload.sh

COPY ./ /app/

WORKDIR /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN python -m pip install "uvicorn[standard]" gunicorn
RUN python -m pip install -r requirements.txt --no-cache-dir

# For development, Jupyter remote kernel, Hydrogen
# Using inside the container:
# jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
ARG INSTALL_JUPYTER=false
RUN bash -c "if [ $INSTALL_JUPYTER == 'true' ] ; then pip install jupyterlab ; fi"

# allow container to wait for other services
ENV WAITFORIT_VERSION="v2.4.1"
RUN curl -o /usr/local/bin/waitforit -sSL https://github.com/maxcnunes/waitforit/releases/download/$WAITFORIT_VERSION/waitforit-linux_amd64 && \
    chmod +x /usr/local/bin/waitforit

ENV PYTHONPATH=/app

FROM backend as celery-worker

ENV C_FORCE_ROOT=1
RUN python -m pip install watchdog pyyaml argh

COPY ./worker-start.sh /worker-start.sh
RUN chmod +x /worker-start.sh
CMD ["bash", "/worker-start.sh"]
