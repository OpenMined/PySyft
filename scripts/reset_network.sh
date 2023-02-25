#!/bin/bash

# WARNING: this will reset all the network VPN keys and clear the route table

# reset headscale keys
HEADSCALE_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 headscale)
COMMAND="rm data/*.* && pgrep headscale | xargs kill -9"
echo $COMMAND | docker exec -i $HEADSCALE_CONTAINER_NAME ash 2>&1

TAILSCALE_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 tailscale)
COMMAND="tailscale down && rm -rf /var/lib/tailscale && pgrep tailscale | xargs kill -9"
echo $COMMAND | docker exec -i $TAILSCALE_CONTAINER_NAME ash 2>&1

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
