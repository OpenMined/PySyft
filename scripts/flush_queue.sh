#!/bin/sh
docker ps --format '{{.Names}}' | grep "celeryworker" |  xargs -I '{}' docker exec -i {} python -c "from grid.core.celery_app import celery_app; celery_app.control.purge();print('Tasks Cleared')"
