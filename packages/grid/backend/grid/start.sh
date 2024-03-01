#! /usr/bin/env bash
set -e

echo "Running start.sh with RELEASE=${RELEASE} and $(id)"
export GEVENT_MONKEYPATCH="False"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
NODE_TYPE=${NODE_TYPE:-domain}
APPDIR=${APPDIR:-$HOME/app}

RELOAD=""
DEBUG_CMD=""

# For debugging permissions
ls -lisa $HOME/data
ls -lisa $APPDIR/syft/
ls -lisa $APPDIR/grid/

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
    pip install --user -e "$APPDIR/syft[telemetry,data_science]"
fi

# only set by kubernetes to avoid conflict with docker tests
if [[ ${DEBUGGER_ENABLED} == "True" ]];
then
    pip install --user debugpy
    DEBUG_CMD="python -m debugpy --listen 0.0.0.0:5678 -m"
fi

set +e
export NODE_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export NODE_UID=$(python $APPDIR/grid/bootstrap.py --uid)
export NODE_TYPE=$NODE_TYPE
set -e

echo "NODE_UID=$NODE_UID"
echo "NODE_TYPE=$NODE_TYPE"

exec $DEBUG_CMD uvicorn $RELOAD --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"
