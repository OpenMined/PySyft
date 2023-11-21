#!/bin/bash

MONGO_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 mongo)
DROPCMD="<<EOF
use app;
db.dropDatabase();
EOF"

FLUSH_COMMAND="mongosh -u root -p example $DROPCMD"
echo "$FLUSH_COMMAND" | docker exec -i $MONGO_CONTAINER_NAME bash 2>&1

# flush the worker queue
. ${BASH_SOURCE%/*}/flush_queue.sh

# reset docker service to clear out weird network issues
sudo service docker restart

# make sure all containers start
. ${BASH_SOURCE%/*}/../packages/grid/scripts/containers.sh
