#!/bin/bash

if [[ -z "$1" ]]; then
    # list db and redis instances
    docker ps --format '{{.Names}}' | grep "db\|redis" | xargs -I '{}' bash -c 'echo "{} -> $(python -c "import sys; print(sys.argv[1].split(\":\")[-1])" "$(docker port {})")"' - {}
else
    PORT=$1
    if docker ps | grep ":${PORT}" | grep -q 'redis'; then
        open redis://127.0.0.1:${PORT}
    else
        open postgresql://postgres:changethis@127.0.0.1:${PORT}/app
    fi
fi
