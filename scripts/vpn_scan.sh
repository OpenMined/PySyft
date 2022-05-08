#!/bin/bash

# use with:
# $ cat scripts/vpn_scan.sh | docker exec -i test_network_1-tailscale-1 ash

apk add netcat-openbsd
apk add lsof
apk add jq
# lsof -i -P -n | grep LISTEN | grep -Eo ':(.+?)\s' | cut -c 2-

# scan local ports
# uncomment to test against localhost
# tailscale status | grep "$(hostname)" | awk '{print $1}' | while read line; do
#     echo "Scanning $line"
#     nc -z -v "$line" 1-65535 2>&1 | grep succeeded
# done

# scan other VPN IPs, make sure to connect the containers to the other networks
# otherwise they will route all the traffic through a relay and this will be really slow
# for example:
# docker network connect test_domain_1_default test_network_1-tailscale-1
# | tr '_' - replaces all underscores with - because tailscale does that now
tailscale status | grep -v "$(hostname | tr '_' -)" | awk '{print $1}' | while read line; do
    echo "Scanning $line"
    if [[ $line =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        nc -z -v "$line" 1-65535 2>&1 | grep succeeded
        # just run one for now
        break
    fi
    # nc -z -v "$line" 21 80 4000 8001 8011 8080 5050 5432 5555 5672 15672 2>&1 | grep succeeded
done
