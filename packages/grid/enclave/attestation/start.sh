#!/usr/bin/env bash
set -e
export PATH="/root/.local/bin:${PATH}"

APP_MODULE=server.attestation_main:app
APP_LOG_LEVEL=${APP_LOG_LEVEL:-info}
UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL:-info}
SERVER_IPV6=${SERVER_IPV6:-"False"}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-4455}
RELOAD=""

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
fi

if [[ ${SERVER_IPV6} == "True" ]];
then
    HOST=${HOST:-"::"}
fi

exec uvicorn $RELOAD --host $HOST --port $PORT --log-level $UVICORN_LOG_LEVEL "$APP_MODULE"
