#!/bin/bash

# script to get notebook URLs
for i in $(docker ps --format "table {{.Names}}" | grep business-logic); do
    port=$(docker port "$i" 8888/tcp | cut -d":" -f2);
    token=$(docker logs "$i" 2>&1 | grep "127.0.0.1:8888/" | cut -d"=" -f2 | head -1);
    echo "$i - http://localhost:$port/?token=$token";
done
