#!/usr/bin/env bash
set -e
export PATH="/root/.local/bin:${PATH}"

APP_MODULE=server.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-4000}
RELOAD=""
VEILID_FLAGS=""

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
    VEILID_FLAGS="--debug"
fi

/usr/bin/veilid-server -c /veilid/veilid-server.conf  $VEILID_FLAGS &

exec uvicorn $RELOAD --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"


