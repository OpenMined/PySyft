#!/bin/bash

NGROK_NAME=${NGROK_NAME:-ngrok}

echo "Using ngrok endpoint [$NGROK_NAME]"

AGENT_ENDPOINT=null
while [ -z "$AGENT_ENDPOINT" ] || [ "$AGENT_ENDPOINT" = "null" ]
do
    echo "Fetching endpoint from ngrok service"
    AGENT_ENDPOINT=$(curl --silent $NGROK_NAME:4040/api/tunnels | ./jq -r '.tunnels[0].public_url')
    echo "ngrok endpoint [$AGENT_ENDPOINT]"
    if [ -z "$AGENT_ENDPOINT" ] || [ "$AGENT_ENDPOINT" = "null" ]; then
        echo "ngrok not ready, sleeping 5 seconds...."
        sleep 5
    fi
done

echo "Fetched endpoint [$AGENT_ENDPOINT]"

export AGENT_ENDPOINT=$AGENT_ENDPOINT
exec "$@"
