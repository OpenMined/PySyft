#!/bin/bash

# find IP of tailscale container
get_tailescale_ip() {
    IP=$(hostname -i)
    IP_RANGE=$(echo ${IP%.*})
    for i in {1..255}; do
        IP_ADDRESS=$IP_RANGE.$i
        OUTPUT=$(nslookup $IP_ADDRESS | grep tailscale)
        if [ ! -z "$OUTPUT" ]; then
            break
        fi
        IP_ADDRESS=""
    done
    echo $IP_ADDRESS
}

set_tailscale_route() {
    TAILSCALE_CONTAINER_IP=$1
    TAILSCALE_CIDR=100.64.0.0/24
    # remove existing route
    ip route del $TAILSCALE_CIDR || true
    # add new route
    ip route add $TAILSCALE_CIDR via $TAILSCALE_CONTAINER_IP dev eth0 onlink
}

TAILSCALE_CONTAINER_IP=$(get_tailescale_ip)
set_tailscale_route $TAILSCALE_CONTAINER_IP
