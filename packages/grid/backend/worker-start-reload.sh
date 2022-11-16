#! /usr/bin/env bash

echo "Running worker-start-reload.sh with RELEASE=${RELEASE}"

set -e

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
else
    echo "There is no script $PRE_START_PATH"
fi

celery -A grid.worker beat -l info --detach
watchmedo auto-restart --directory=/app --pattern=*.py --recursive -- celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
