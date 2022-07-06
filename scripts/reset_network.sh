#!/bin/bash

# WARNING: this will reset all the network VPN keys and clear the route table

# reset headscale keys
HEADSCALE_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 headscale)
COMMAND="rm data/*.* && pgrep headscale | xargs kill -9"
echo $COMMAND | docker exec -i $HEADSCALE_CONTAINER_NAME ash 2>&1

TAILSCALE_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 tailscale)
COMMAND="tailscale down && rm -rf /var/lib/tailscale && pgrep tailscale | xargs kill -9"
echo $COMMAND | docker exec -i $TAILSCALE_CONTAINER_NAME ash 2>&1

# reset node and node_route tables in db
DB_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 db)
FLUSH_COMMAND='psql -U postgres -d app -c "TRUNCATE node, node_route, association_request RESTART IDENTITY;"'
echo $FLUSH_COMMAND | docker exec -i $DB_CONTAINER_NAME bash 2>&1

# flush the worker queue
. ${BASH_SOURCE%/*}/flush_queue.sh

# reset docker service to clear out weird network issues
sudo service docker restart

# make sure all containers start
. ${BASH_SOURCE%/*}/../packages/grid/scripts/containers.sh
