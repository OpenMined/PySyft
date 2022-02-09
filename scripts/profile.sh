#!/bin/bash

# install ps
docker ps --format '{{.Names}}' | grep "backend\|celery" | xargs -I {} docker exec {} bash -c "apt-get update && apt-get install procps -y"

# start the workers
docker ps --format '{{.Names}}' | grep "backend" | xargs -I {} docker exec {} /start-reload.sh &
docker ps --format '{{.Names}}' | grep "celery" | xargs -I {} docker exec {} /worker-start-reload.sh &

# run the integration test with scalene on client side
scalene --json --outfile packages/grid/profile/client.json tests/integration/e2e/spicy_bird_performance_test.py

# kill all the python processes to flush the logs
docker ps --format '{{.Names}}' | grep "backend\|celery" | xargs -I {} docker exec {} bash -c "pgrep python | xargs kill -9"
