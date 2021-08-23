#! /usr/bin/env bash
set -e

pip install -e /app/syft

python /app/app/celeryworker_pre_start.py

watchmedo auto-restart --directory=/app --pattern=*.py --recursive -- celery worker -A app.worker -l info -Q main-queue -c 1
