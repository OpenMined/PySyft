#!/bin/bash

# WARNING: this will drop the app database in all your mongo dbs
echo $1

if [ -z $1 ]; then
    MONGO_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -m 1 mongo)
else
    MONGO_CONTAINER_NAME=$1
fi

DROPCMD="<<EOF
use app;
db.dropDatabase();
EOF"

FLUSH_COMMAND="mongosh -u root -p example $DROPCMD"
echo "$FLUSH_COMMAND" | docker exec -i $MONGO_CONTAINER_NAME bash 2>&1