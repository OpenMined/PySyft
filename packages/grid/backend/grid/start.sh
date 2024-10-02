#! /usr/bin/env bash
set -e

echo "Running Syft with RELEASE=${RELEASE}"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
APPDIR=${APPDIR:-$HOME/app}
RELOAD=""
ROOT_PROC=""

export CREDENTIALS_PATH=${CREDENTIALS_PATH:-$HOME/data/creds/credentials.json}
export SERVER_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export SERVER_UID=$(python $APPDIR/grid/bootstrap.py --uid)

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
    # TODO: Polish these values up
    DEPLOYMENT_ENV="$SERVER_TYPE-$SERVER_SIDE_TYPE"
    RESOURCE_ATTRS=(
        "deployment.environment=$DEPLOYMENT_ENV"
        "service.namespace=$DEPLOYMENT_ENV"
        "service.instance.id=$SERVER_UID"
        "k8s.pod.name=${K8S_POD_NAME:-"none"}"
        "k8s.namespace.name=${K8S_NAMESPACE:"none"}"
        "syft.server.uid=$SERVER_UID"
        "syft.server.type=$SERVER_TYPE"
        "syft.server.side.type=$SERVER_SIDE_TYPE"
    )

    # environ is always prefixed with the server type
    export OTEL_SERVICE_NAME="${DEPLOYMENT_ENV}-${OTEL_SERVICE_NAME:-"backend"}"
    export OTEL_RESOURCE_ATTRIBUTES=$(IFS=, ; echo "${RESOURCE_ATTRS[*]}")

    echo "OpenTelemetry Enabled"
    env | grep OTEL_
else
    echo "OpenTelemetry Disabled"
fi

echo "SERVER_UID=$SERVER_UID"
echo "SERVER_TYPE=$SERVER_TYPE"
echo "SERVER_SIDE_TYPE=$SERVER_SIDE_TYPE"

exec $ROOT_PROC uvicorn $RELOAD --host $HOST --port $PORT --log-config=$APPDIR/grid/logging.yaml "$APP_MODULE"
