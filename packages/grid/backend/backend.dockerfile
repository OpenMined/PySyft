FROM python:3.9.6-slim as build

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends curl wget

COPY ./ /app/

WORKDIR /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN pip install --user "uvicorn[standard]" gunicorn
RUN pip install --user -r requirements.txt

# For development, Jupyter remote kernel, Hydrogen
# Using inside the container:
# jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
ARG INSTALL_JUPYTER=false
RUN bash -c "if [ $INSTALL_JUPYTER == 'true' ] ; then pip install --user jupyterlab ; fi"

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

COPY --from=build /app /app
COPY --from=build /root/.local /root/.local
COPY --from=build /usr/local/bin/waitforit /usr/local/bin/waitforit

WORKDIR /app/

# Celery worker
FROM backend as celery-worker
ENV C_FORCE_ROOT=1
RUN pip install --user watchdog pyyaml argh

COPY ./worker-start.sh /worker-start.sh
RUN chmod +x /worker-start.sh
CMD ["bash", "/worker-start.sh"]
