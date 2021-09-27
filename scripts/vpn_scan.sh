#!/bin/sh

# use with:
# $ cat scripts/vpn_scan.sh | docker exec -i test_network_1_tailscale_1 ash

apk add netcat-openbsd
apk add lsof
# lsof -i -P -n | grep LISTEN | grep -Eo ':(.+?)\s' | cut -c 2-

# scan local ports
tailscale status | grep "$(hostname)" | awk '{print $1}' | while read line; do
    echo "Scanning $line"
    nc -z -v "$line" 1-65535 2>&1 | grep succeeded
done

# scan other VPN IPs, this is slow so we are only including known ports for now
tailscale status | grep -v "$(hostname)" | awk '{print $1}' | while read line; do
    echo "Scanning $line"
    nc -z -v "$line" 21 80 4000 8001 8011 8080 5050 5432 5555 5672 15672 41641 2>&1 | grep succeeded
done
