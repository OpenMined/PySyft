#! /usr/bin/env bash
set -e

echo "Running start.sh with RELEASE=${RELEASE}"

RELOAD=""
if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
    pip install -e "/app/syft[telemetry,data_science]"
fi

export RUST_BACKTRACE=$RUST_BACKTRACE

set +e
export NODE_PRIVATE_KEY=$(python bootstrap.py --private_key)
export NODE_UID=$(python bootstrap.py --uid)
set -e

echo "NODE_UID=$NODE_UID"
echo "NODE_TYPE=$NODE_TYPE"

APP_MODULE=worker:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

exec uvicorn $RELOAD --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"
