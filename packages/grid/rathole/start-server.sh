#!/usr/bin/env bash

APP_MODULE=server.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${APP_PORT:-5555}
RELOAD="--reload"
DEBUG_CMD=""

apt update && apt install -y nginx
nginx &
exec python -m $DEBUG_CMD uvicorn $RELOAD --host $HOST --port $APP_PORT --log-level $LOG_LEVEL "$APP_MODULE" &
/app/rathole server.toml
