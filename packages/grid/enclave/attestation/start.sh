#!/usr/bin/env bash
set -e
export PATH="/root/.local/bin:${PATH}"

APP_MODULE=server.attestation_main:app
APP_LOG_LEVEL=${APP_LOG_LEVEL:-info}
HYPERCORN_LOG_LEVEL=${HYPERCORN_LOG_LEVEL:-info}
PORT=${PORT:-4455}
RELOAD=""

if [[ ${DEV_MODE} == "True" ]];
then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
fi

# Check for IPv4 and IPv6 availability
IPV4_AVAILABLE=$(grep -c -v ":" /proc/net/if_inet6)
IPV6_AVAILABLE=$(grep -c ":" /proc/net/if_inet6)

# Default to IPv4+IPv6 if no preference is set
HOST_MODE=${HOST_MODE:-ipv4}

if [[ $HOST_MODE == "ipv4" ]] && [[ $IPV4_AVAILABLE -gt 0 ]]; then
    # IPv4 is available
    HOST=${HOST:-0.0.0.0}
    exec hypercorn $RELOAD --bind $HOST:$PORT --log-level $HYPERCORN_LOG_LEVEL "$APP_MODULE"
elif [[ $HOST_MODE == "ipv6" ]] && [[ $IPV6_AVAILABLE -gt 0 ]]; then
    # IPv6 is available
    HOST=${HOST:-[::]}
    exec hypercorn $RELOAD --bind $HOST:$PORT --log-level $HYPERCORN_LOG_LEVEL "$APP_MODULE"
elif [[ $HOST_MODE == "ipv4+ipv6" ]] && [[ $IPV4_AVAILABLE -gt 0 ]] && [[ $IPV6_AVAILABLE -gt 0 ]]; then
    # Both IPv4 and IPv6 are available
    BINDINGS="${HOST:-0.0.0.0}:${PORT},${HOST:-[::]}:${PORT}"
    exec hypercorn $RELOAD --bind $BINDINGS --log-level $HYPERCORN_LOG_LEVEL "$APP_MODULE"
else
    echo "No suitable IP version available for the specified HOST_MODE: $HOST_MODE"
    exit 1
fi
