#! /usr/bin/env bash
set -e

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

pip install --user -e /app/syft

python /app/grid/backend_prestart.py

watchmedo auto-restart --directory=/app --pattern=*.py --recursive -- celery worker -A grid.worker -l info -Q main-queue -c 1
