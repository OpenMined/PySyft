#! /usr/bin/env bash
set -e

echo "Running Syft with RELEASE=${RELEASE}"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
SERVER_TYPE=${SERVER_TYPE:-datasite}
APPDIR=${APPDIR:-$HOME/app}
RELOAD=""
ROOT_PROC=""

if [[ ${DEV_MODE} == "True" ]];
then
    echo "Hot-reload Enabled"
    RELOAD="--reload"
fi

# only set by kubernetes to avoid conflict with docker tests
if [[ ${DEBUGGER_ENABLED} == "True" ]];
then
    echo "Debugger Enabled"
    uv pip install debugpy
    ROOT_PROC="python -m debugpy --listen 0.0.0.0:5678 -m"
fi

if [[ ${TRACING} == "true" ]];
then
    # TODOs:
    # ? Kubernetes OTel operator is recommended by signoz
    export OTEL_PYTHON_LOG_CORRELATION=${OTEL_PYTHON_LOG_CORRELATION:-true}
    echo "OpenTelemetry Enabled. Endpoint=$OTEL_EXPORTER_OTLP_ENDPOINT Protocol=$OTEL_EXPORTER_OTLP_PROTOCOL"
else
    echo "OpenTelemetry Disabled"
fi

export CREDENTIALS_PATH=${CREDENTIALS_PATH:-$HOME/data/creds/credentials.json}
export SERVER_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export SERVER_UID=$(python $APPDIR/grid/bootstrap.py --uid)
export SERVER_TYPE=$SERVER_TYPE

echo "SERVER_UID=$SERVER_UID"
echo "SERVER_TYPE=$SERVER_TYPE"

exec $ROOT_PROC uvicorn $RELOAD --host $HOST --port $PORT --log-config=$APPDIR/grid/logging.yaml "$APP_MODULE"
