#!/bin/bash

# Detect if IPv4 is available
if ip -4 addr show scope global | grep -q inet; then
    IPV4_AVAILABLE=true
else
    IPV4_AVAILABLE=false
fi

# Detect if IPv6 is available
if ip -6 addr show scope global | grep -q inet6; then
    IPV6_AVAILABLE=true
else
    IPV6_AVAILABLE=false
fi

# Start hypercorn with the appropriate options
if [ "$IPV4_AVAILABLE" = true ] && [ "$IPV6_AVAILABLE" = true ]; then
    exec hypercorn --bind 0.0.0.0:$MOUNT_API_PORT --bind [::]:$MOUNT_API_PORT --log-config logging.yaml src.api:app
elif [ "$IPV4_AVAILABLE" = true ]; then
    exec hypercorn --bind 0.0.0.0:$MOUNT_API_PORT --log-config logging.yaml src.api:app
elif [ "$IPV6_AVAILABLE" = true ]; then
    exec hypercorn --bind [::]:$MOUNT_API_PORT --log-config logging.yaml src.api:app
else
    echo "Neither IPv4 nor IPv6 is available. Exiting."
    exit 1
fi
