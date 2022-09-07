#!/bin/bash
CMD=""
if [[ -z "$1" ]]; then
    # list db, redis, rabbitmq, seaweedfs, jaeger, mongo ports
    docker ps --format '{{.Names}}' | grep "db\|redis\|queue\|seaweedfs\|jaeger\|mongo" | xargs -I '{}' bash -c 'echo "{} -> $(python3 -c "import sys; [print(l.split(\":\")[-1]) for l in sys.argv[1].split(\"\n\") if l.split(\"/tcp\")[0] in sys.argv[2].split(\",\")]" "$(docker port {})" "15672,5432,6379,8888,16686,27017")"' - {}
else
    PORT=$1
    if docker ps | grep ":${PORT}" | grep -q 'redis'; then
        CMD="redis://127.0.0.1:${PORT}"
    elif docker ps | grep ":${PORT}" | grep -q 'postgres'; then
        CMD="postgresql://postgres:changethis@127.0.0.1:${PORT}/app"
    elif docker ps | grep ":${PORT}" | grep -q 'mongo'; then
        CMD="mongodb://root:example@127.0.0.1:${PORT}"
    else
        CMD="http://localhost:${PORT}"
    fi

    if [[ -n "$IS_WSL" || -n "$WSL_DISTRO_NAME" ]]; then
        cmd.exe /c start $CMD
    elif [[ $OSTYPE == 'darwin'* ]]; then
        open $CMD
    else
        echo "Please add linux support"
    fi
fi
