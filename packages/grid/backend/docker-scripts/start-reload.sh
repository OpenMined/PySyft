#! /usr/bin/env sh
set -e

echo "Running start-reload.sh with RELEASE=${RELEASE}"

python3 -c "print('---Monkey Patching: Gevent---\n');from gevent import monkey;monkey.patch_all()"

if [ -f /app/grid/main.py ]; then
    DEFAULT_MODULE_NAME=grid.main
    elif [ -f /app/main.py ]; then
    DEFAULT_MODULE_NAME=main
fi

echo "Using $DEFAULT_MODULE_NAME"

MODULE_NAME=${MODULE_NAME:-$DEFAULT_MODULE_NAME}
VARIABLE_NAME=${VARIABLE_NAME:-app}
export APP_MODULE=${APP_MODULE:-"$MODULE_NAME:$VARIABLE_NAME"}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
LOG_LEVEL=${LOG_LEVEL:-info}

if [ ${TAILSCALE_ROUTE_VIA_POD-} = "enabled" ]; then
  tailscale_ip=$(kubectl get pod -n openmined tailscale-0 -o jsonpath='{.status.podIP}')
  ip route add 100.64.0.0/10 via "${tailscale_ip}"
  echo "Routing tailnet via ${tailscale_ip}"
fi

# If there's a prestart.sh script in the /app directory or other path specified, run it before starting
PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
else
    echo "There is no script $PRE_START_PATH"
fi

# Start Uvicorn with live reload
exec uvicorn --reload --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"

## fetched from https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/start.sh
