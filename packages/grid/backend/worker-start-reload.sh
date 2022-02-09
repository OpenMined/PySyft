#! /usr/bin/env bash

echo "Running worker-start-reload.sh with RELEASE=${RELEASE}"

set -e

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

pip install --user -e /app/syft[dev]

python3 -c "print('---Monkey Patching: Gevent---\n');from gevent import monkey;monkey.patch_all()"
python /app/grid/backend_prestart.py

watchmedo auto-restart --directory=/app --pattern=*.py --recursive -- celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
