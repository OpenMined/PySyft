#! /usr/bin/env bash
set -e

echo "Running Syft with RELEASE=${RELEASE} and $(id)"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
NODE_TYPE=${NODE_TYPE:-domain}
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
export NODE_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export NODE_UID=$(python $APPDIR/grid/bootstrap.py --uid)
export NODE_TYPE=$NODE_TYPE

echo "NODE_UID=$NODE_UID"
echo "NODE_TYPE=$NODE_TYPE"

exec $DEBUG_CMD uvicorn $RELOAD --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"
