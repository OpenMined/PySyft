#! /usr/bin/env bash

echo "Running worker-start.sh with RELEASE=${RELEASE}"

set -e

python /app/grid/backend_prestart.py

celery -A grid.periodic_tasks beat -l info --detach && celery -A grid.periodic_tasks worker -l info -Q celery -n beatworker.%h
