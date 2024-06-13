#!/bin/bash

# Détecter si IPv4 est disponible
if ip -4 addr show scope global | grep -q inet; then
    IPV4_AVAILABLE=true
else
    IPV4_AVAILABLE=false
fi

# Détecter si IPv6 est disponible
if ip -6 addr show scope global | grep -q inet6; then
    IPV6_AVAILABLE=true
else
    IPV6_AVAILABLE=false
fi

# Démarrer uvicorn avec les options appropriées
if [ "$IPV4_AVAILABLE" = true ] && [ "$IPV6_AVAILABLE" = true ]; then
    exec uvicorn --host=0.0.0.0 --host=:: --port=$MOUNT_API_PORT --log-config=logging.yaml src.api:app
elif [ "$IPV4_AVAILABLE" = true ]; then
    exec uvicorn --host=0.0.0.0 --port=$MOUNT_API_PORT --log-config=logging.yaml src.api:app
elif [ "$IPV6_AVAILABLE" = true ]; then
    exec uvicorn --host=:: --port=$MOUNT_API_PORT --log-config=logging.yaml src.api:app
else
    echo "Neither IPv4 nor IPv6 is available. Exiting."
    exit 1
fi
