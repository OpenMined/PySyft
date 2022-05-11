#!/bin/bash

# WARNING: this will reset all the network VPN keys and clear the route table

# reset headscale keys
TAILSCALE_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 headscale)
COMMAND="rm data/*.* && pgrep headscale | xargs kill -9"
echo $COMMAND | docker exec -i $TAILSCALE_CONTAINER_NAME ash 2>&1

# reset node and node_route tables in db
DB_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 db)
FLUSH_COMMAND='psql -U postgres -d app -c "TRUNCATE node, node_route RESTART IDENTITY;"'
echo $FLUSH_COMMAND | docker exec -i $DB_CONTAINER_NAME bash 2>&1
