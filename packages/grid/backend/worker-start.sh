#! /usr/bin/env bash

echo "Running worker-start.sh with RELEASE=${RELEASE}"

set -e

PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
else
    echo "There is no script $PRE_START_PATH"
fi

celery -A grid.worker beat -l info --detach && celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
