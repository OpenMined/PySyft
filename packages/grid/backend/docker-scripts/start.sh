#! /usr/bin/env sh
set -e

echo "Running start.sh with RELEASE=${RELEASE}"

if [ -f /app/grid/main.py ]; then
    DEFAULT_MODULE_NAME=grid.main
    elif [ -f /app/main.py ]; then
    DEFAULT_MODULE_NAME=main
fi
MODULE_NAME=${MODULE_NAME:-$DEFAULT_MODULE_NAME}
VARIABLE_NAME=${VARIABLE_NAME:-app}
export APP_MODULE=${APP_MODULE:-"$MODULE_NAME:$VARIABLE_NAME"}

# if [ -f /app/gunicorn_conf.py ]; then
#     DEFAULT_GUNICORN_CONF=/app/gunicorn_conf.py
#     elif [ -f /app/grid/gunicorn_conf.py ]; then
#     DEFAULT_GUNICORN_CONF=/app/grid/gunicorn_conf.py
# else
#     DEFAULT_GUNICORN_CONF=/gunicorn_conf.py
# fi
# export GUNICORN_CONF=${GUNICORN_CONF:-$DEFAULT_GUNICORN_CONF}
# export WORKER_CLASS=${WORKER_CLASS:-"uvicorn.workers.UvicornWorker"}

# If there's a prestart.sh script in the /app directory or other path specified, run it before starting
PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
else
    echo "There is no script $PRE_START_PATH"
fi

# if we run on kubernetes add the tailscale container as a gateway for 100.64.0.0/24
if [ "$CONTAINER_HOST" == "kubernetes" ]; then
    . tailscale-gateway.sh
fi

# Start Gunicorn
# TODO: gunicorn crashes when running in k8s with asyncio issues while uvicorn from
# start-reload.sh seems okay
# exec gunicorn -k "$WORKER_CLASS" -c "$GUNICORN_CONF" "$APP_MODULE"

LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
exec uvicorn --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"

## fetched from https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/start.sh
