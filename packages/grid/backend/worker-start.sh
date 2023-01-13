#! /usr/bin/env bash

echo "Running worker-start.sh with RELEASE=${RELEASE}"

set -e

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

set +e
NODE_PRIVATE_KEY=$(python /app/grid/bootstrap.py --private_key)
NODE_UID=$(python /app/grid/bootstrap.py --uid)
set -e

echo "NODE_PRIVATE_KEY=$NODE_PRIVATE_KEY"
echo "NODE_UID=$NODE_UID"

export NODE_UID=$NODE_UID
export NODE_PRIVATE_KEY=$NODE_PRIVATE_KEY


if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="watchmedo auto-restart --directory=/app --pattern=*.py --recursive --"
fi

celery -A grid.worker beat -l info --detach && \
$RELOAD celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500