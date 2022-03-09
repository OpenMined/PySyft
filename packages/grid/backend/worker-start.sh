#! /usr/bin/env bash

echo "Running worker-start.sh with RELEASE=${RELEASE}"

set -e

python3 -c "print('---Monkey Patching: Gevent---\n');from gevent import monkey;monkey.patch_all()"
python /app/grid/backend_prestart.py

celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
