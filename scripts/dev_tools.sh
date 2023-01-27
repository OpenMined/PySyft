#!/bin/bash

unameOut="$(uname -rs)"
case "${unameOut}" in
    # WSL
    Linux*Microsoft*WSL)  command="echo";;
    # Win Git Bash
    CYGWIN*|MINGW*|MSYS*) command="start";;
    # Linux + flavors
    Linux*)               command="xdg-open";;
    # macOS
    Darwin*)              command="open";;
    *)                    command="echo UNKNOWN OS: ${unameOut} | "
esac

function docker_list_exposed_ports() {
    FILTER=$1
    echo "-----------------------------"
    echo "Containers | Exposed Ports"
    echo "-----------------------------"
    docker ps --format '{{.Names}}' | grep $FILTER | xargs docker inspect --format='{{.Name}}  {{range $port, $portMap := .NetworkSettings.Ports}}{{if $portMap }}{{(index $portMap 0).HostPort}} {{end}}{{end}}'
}

if [[ -z "$1" ]]; then
    # list db, redis, rabbitmq, and seaweedfs ports
    docker_list_exposed_ports "db\|redis\|queue\|seaweedfs\|jaeger\|mongo"
else
    PORT=$1
    if docker ps | grep ":${PORT}" | grep -q 'redis'; then
        ${command} redis://127.0.0.1:${PORT}
    elif docker ps | grep ":${PORT}" | grep -q 'postgres'; then
        ${command} postgresql://postgres:changethis@127.0.0.1:${PORT}/app
    elif docker ps | grep ":${PORT}" | grep -q 'mongo'; then
        ${command} mongodb://root:example@127.0.0.1:${PORT}
    else
        ${command} http://localhost:${PORT}
    fi
fi
