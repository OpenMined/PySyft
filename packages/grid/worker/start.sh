#! /usr/bin/env bash
set -e

echo "Running start.sh with RELEASE=${RELEASE}"

RELOAD=""
# RUST_BACKTRACE=0
if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    # RELOAD="--dev"
    # RUST_BACKTRACE="full"
    RELOAD="--reload"
fi

CREDENTIALS_PATH="./credentials.json"
export CREDENTIALS_PATH=$CREDENTIALS_PATH
export RUST_BACKTRACE=$RUST_BACKTRACE

set +e
NODE_PRIVATE_KEY=$(python ./bootstrap.py --private_key)
NODE_UID=$(python ./bootstrap.py --uid)
set -e

echo "NODE_PRIVATE_KEY=$NODE_PRIVATE_KEY"
echo "NODE_UID=$NODE_UID"

export NODE_UID=$NODE_UID
export NODE_PRIVATE_KEY=$NODE_PRIVATE_KEY

# add back robyn
# python worker.py $RELOAD

APP_MODULE=worker:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

exec uvicorn $RELOAD --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"
