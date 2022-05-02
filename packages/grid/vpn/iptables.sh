#!/bin/bash

# find IP of tailscale container
get_tailescale_ip_from_container_net() {
    IP_RANGES=$(get_possible_ips)
    echo "Searching these ip ranges: \n$IP_RANGES" >&2
    for IP_RANGE in $IP_RANGES
    do
        for i in {1..255}; do
            IP_ADDRESS=$IP_RANGE.$i
            echo "nslookup: $IP_ADDRESS" >&2
            OUTPUT=$(nslookup $IP_ADDRESS | grep tailscale)
            if [ ! -z "$OUTPUT" ]; then
                echo "Found: hostname tailscale on $IP_ADDRESS" >&2
                break 2
            fi
            IP_ADDRESS=""
        done
    done
    echo $IP_ADDRESS
}

# find IP of tailscale container
get_tailescale_internal_ip_from_host_net() {
    IP_RANGES=$(get_possible_ips)
    echo "Searching these ip ranges: \n$IP_RANGES" >&2
    for IP_RANGE in $IP_RANGES
    do
        for i in {1..255}; do
            IP_ADDRESS=$IP_RANGE.$i
            echo "curl: $IP_ADDRESS" >&2
            OUTPUT=$(curl -s --max-time 0.01 $IP_ADDRESS:8082/ping > /dev/null)
            TRAEFIK_HEALTH_OK=$?
            if [ "$TRAEFIK_HEALTH_OK" == "0" ]; then
                echo "Found: traefik ping on $IP_ADDRESS" >&2
                break 2
            fi
            IP_ADDRESS=""
        done
    done
    echo $IP_ADDRESS
}

get_device() {
    IP=$1
    IP_RANGE=$(echo ${IP%.*})
    IFACE=$(ip route | grep -v default | grep -m 1 $IP_RANGE | awk '{ print $3 }')
    echo $IFACE
}

get_possible_ips() {
    IP_RANGES=$(ip route | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | sort | uniq)
    echo $IP_RANGES
}

set_tailscale_route() {
    TAILSCALE_CONTAINER_IP=$1
    TAILSCALE_CIDR=100.64.0.0/24
    # remove existing route
    ip route del $TAILSCALE_CIDR || true
    # add new route
    IFACE=$(get_device $TAILSCALE_CONTAINER_IP)
    echo "Adding route with: ip route add $TAILSCALE_CIDR via $TAILSCALE_CONTAINER_IP dev $IFACE onlink"
    ip route add $TAILSCALE_CIDR via $TAILSCALE_CONTAINER_IP dev $IFACE onlink
}

TAILSCALE_CONTAINER_IP=""

while [ -z "$TAILSCALE_CONTAINER_IP" ]
do
    echo "Searching for TAILSCALE_CONTAINER_IP"
    TAILSCALE_CONTAINER_IP=$(get_tailescale_internal_ip_from_host_net)
    if [ -z "$TAILSCALE_CONTAINER_IP" ]; then
        echo "Using backup nslookup method"
        TAILSCALE_CONTAINER_IP=$(get_tailescale_ip_from_container_net)
    fi
    if [ -z "$TAILSCALE_CONTAINER_IP" ]; then
        echo "Failed to find TAILSCALE_CONTAINER_IP=$TAILSCALE_CONTAINER_IP"
    else
        echo "Found TAILSCALE_CONTAINER_IP=$TAILSCALE_CONTAINER_IP"
        set_tailscale_route $TAILSCALE_CONTAINER_IP
    fi
done

sleep infinity
