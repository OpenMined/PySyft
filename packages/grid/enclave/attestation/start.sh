#!/usr/bin/env bash
set -e
export PATH="/root/.local/bin:${PATH}"

APP_MODULE=server.attestation_main:app
APP_LOG_LEVEL=${APP_LOG_LEVEL:-info}
HYPERCORN_LOG_LEVEL=${HYPERCORN_LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-4455}
RELOAD=""

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
fi


exec hypercorn $RELOAD --bind $HOST:$PORT --log-level $HYPERCORN_LOG_LEVEL "$APP_MODULE"
