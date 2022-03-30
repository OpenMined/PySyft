#!/bin/bash

# unfortunately sometimes the containers are not running after reboot
# this script makes sure they are restarted
pidof -o %PPID -x $0 >/dev/null && echo "ERROR: Script $0 already running" && exit 1

until systemctl is-active --quiet docker
do
    echo "Waiting for docker service to start"
    sleep 1
done

while [ $(docker ps --filter "status=exited" -q | wc -l) != "0" ]; do
    # the following containers are not running
    echo "CRON: List of containers that are not running:"
    docker ps --filter "status=exited" --format "{{.Names}} {{.Status}}" | xargs -I {} echo {}
    echo "CRON: Forcing containers to start"
    docker ps --filter "status=exited" -q | xargs -I {} docker start {}
    sleep 1
done
