#!/bin/bash

NGROK_NAME=${NGROK_NAME:-ngrok}

echo "ngrok end point [$NGROK_NAME]"

AGENT_ENDPOINT=null
while [ -z "$AGENT_ENDPOINT" ] || [ "$AGENT_ENDPOINT" = "null" ]
do
    echo "Fetching end point from ngrok service"
    AGENT_ENDPOINT=$(curl --silent $NGROK_NAME:4040/api/tunnels | ./jq -r '.tunnels[0].public_url')
    echo "ngrok end point [$AGENT_ENDPOINT]"
    if [ -z "$AGENT_ENDPOINT" ] || [ "$AGENT_ENDPOINT" = "null" ]; then
        echo "ngrok not ready, sleeping 5 seconds...."
        sleep 5
    fi
done

echo "fetched end point [$AGENT_ENDPOINT]"

export AGENT_ENDPOINT=$AGENT_ENDPOINT
exec "$@"