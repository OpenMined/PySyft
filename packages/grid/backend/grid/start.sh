#! /usr/bin/env bash
set -e

echo "Running Syft with RELEASE=${RELEASE}"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
NODE_TYPE=${NODE_TYPE:-domain}
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

if [[ ${TRACING} == "True" ]];
then
    echo "OpenTelemetry Enabled"

    # TODOs:
    # ! Handle case when OTEL_EXPORTER_OTLP_ENDPOINT is not set.
    # ! syft-signoz-otel-collector.platform:4317 should be plumbed through helm charts
    # ? Kubernetes OTel operator is recommended by signoz
    export OTEL_PYTHON_LOG_CORRELATION=${OTEL_PYTHON_LOG_CORRELATION:-true}
    export OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-"http://syft-signoz-otel-collector.platform:4317"}
    export OTEL_EXPORTER_OTLP_PROTOCOL=${OTEL_EXPORTER_OTLP_PROTOCOL:-grpc}

    # TODO: uvicorn postfork is not stable with OpenTelemetry
    # ROOT_PROC="opentelemetry-instrument"
fi

export CREDENTIALS_PATH=${CREDENTIALS_PATH:-$HOME/data/creds/credentials.json}
export NODE_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export NODE_UID=$(python $APPDIR/grid/bootstrap.py --uid)
export NODE_TYPE=$NODE_TYPE

echo "NODE_UID=$NODE_UID"
echo "NODE_TYPE=$NODE_TYPE"

exec $ROOT_PROC uvicorn $RELOAD --host $HOST --port $PORT --log-config=$APPDIR/grid/logging.yaml "$APP_MODULE"
