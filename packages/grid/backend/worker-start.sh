#! /usr/bin/env bash

echo "Running worker-start.sh with RELEASE=${RELEASE}"

set -e

python /app/grid/backend_prestart.py

# if we run on kubernetes add the tailscale container as a gateway for 100.64.0.0/24
if [ "$CONTAINER_HOST" == "kubernetes" ]; then
    . tailscale-gateway.sh
fi

celery -A grid.worker beat -l info --detach && celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
