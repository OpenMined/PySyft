#!/bin/sh

set -e

# Retry until 2 minutes
RETRY_ARGS="--retry 24 --retry-delay 5 --retry-all-errors"
MASTER_URL="localhost:9333"
FILER_URL="localhost:8888"

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

# Select appropriate URLs based on the availability of IPv4 and/or IPv6
if [ "$IPV4_AVAILABLE" = true ] && [ "$IPV6_AVAILABLE" = true ]; then
    MASTER_URLS=("http://localhost:9333" "http://[::1]:9333")
    FILER_URLS=("http://localhost:8888" "http://[::1]:8888")
elif [ "$IPV4_AVAILABLE" = true ]; then
    MASTER_URLS=("http://localhost:9333")
    FILER_URLS=("http://localhost:8888")
elif [ "$IPV6_AVAILABLE" = true ]; then
    MASTER_URLS=("http://[::1]:9333")
    FILER_URLS=("http://[::1]:8888")
else
    echo "Neither IPv4 nor IPv6 is available. Exiting."
    exit 1
fi

# Function to check URLs
check_url() {
    local url=$1
    curl --silent $RETRY_ARGS $url > /dev/null
}

# Check cluster health and volume status
for url in "${MASTER_URLS[@]}"; do
    check_url "$url/cluster/healthz"
    check_url "$url/vol/status"
done

# Check filer URL
for url in "${FILER_URLS[@]}"; do
    check_url "$url/"
done
