#! /usr/bin/env bash
set -e

pip install nltk
echo "Running start.sh with RELEASE=${RELEASE}"
export GEVENT_MONKEYPATCH="False"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
RELOAD=""
NODE_TYPE=${NODE_TYPE:-domain}

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
    # pip install -vvv --no-cache-dir ansible_core==2.15.5
    # pip install -vvv --no-cache-dir ansible==8.5.0
    pip install -e "/app/syft"
    echo "done installing syft in dev mode"
fi

set +e
NODE_PRIVATE_KEY=$(python /app/grid/bootstrap.py --private_key)
NODE_UID=$(python /app/grid/bootstrap.py --uid)
set -e

echo "NODE_PRIVATE_KEY=$NODE_PRIVATE_KEY"
echo "NODE_UID=$NODE_UID"
echo "NODE_TYPE=$NODE_TYPE"

export NODE_UID=$NODE_UID
export NODE_PRIVATE_KEY=$NODE_PRIVATE_KEY
export NODE_TYPE=$NODE_TYPE

# export GEVENT_MONKEYPATCH="True"
exec uvicorn $RELOAD --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE"
