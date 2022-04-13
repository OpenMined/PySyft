#! /usr/bin/env bash

echo "Running worker-start-reload.sh with RELEASE=${RELEASE}"

set -e

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

pip install --user -e /app/syft[dev]

python /app/grid/backend_prestart.py

# if we run on kubernetes add the tailscale container as a gateway for 100.64.0.0/24
if [ "$CONTAINER_HOST" == "kubernetes" ]; then
    . tailscale-gateway.sh
fi

watchmedo auto-restart --directory=/app --pattern=*.py --recursive -- celery -A grid.worker worker -l info -Q main-queue --pool=gevent -c 500
