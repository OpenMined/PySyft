#! /usr/bin/env bash
set -e

echo "Running Syft with RELEASE=${RELEASE} and $(id)"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
SERVER_TYPE=${SERVER_TYPE:-datasite}
APPDIR=${APPDIR:-$HOME/app}
RELOAD=""
DEBUG_CMD=""

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
fi

# only set by kubernetes to avoid conflict with docker tests
if [[ ${DEBUGGER_ENABLED} == "True" ]];
then
    uv pip install debugpy
    DEBUG_CMD="python -m debugpy --listen 0.0.0.0:5678 -m"
fi

export CREDENTIALS_PATH=${CREDENTIALS_PATH:-$HOME/data/creds/credentials.json}
export SERVER_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export SERVER_UID=$(python $APPDIR/grid/bootstrap.py --uid)
export SERVER_TYPE=$SERVER_TYPE

echo "SERVER_UID=$SERVER_UID"
echo "SERVER_TYPE=$SERVER_TYPE"

exec $DEBUG_CMD uvicorn $RELOAD --host $HOST --port $PORT --log-config=$APPDIR/grid/logging.yaml --log-level $LOG_LEVEL "$APP_MODULE"
