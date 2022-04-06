#! /usr/bin/env bash

echo "Running worker-start.sh with RELEASE=${RELEASE}"

set -e

python /app/grid/backend_prestart.py

celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
