#! /usr/bin/env bash
set -e

pip install --user -e /app/syft

python /app/grid/backend_prestart.py

watchmedo auto-restart --directory=/app --pattern=*.py --recursive -- celery worker -A grid.worker -l info -Q main-queue -c 1
