#! /usr/bin/env bash
set -e

echo "Running Syft with RELEASE=${RELEASE} and $(id)"

APP_MODULE=grid.main:app
LOG_LEVEL=${LOG_LEVEL:-info}
PORT=${PORT:-80}
NODE_TYPE=${NODE_TYPE:-domain}
APPDIR=${APPDIR:-$HOME/app}
RELOAD=""
DEBUG_CMD=""

if [[ ${DEV_MODE} == "True" ]]; then
    echo "DEV_MODE Enabled"
    RELOAD="--reload"
fi

# only set by kubernetes to avoid conflict with docker tests
if [[ ${DEBUGGER_ENABLED} == "True" ]]; then
    pip install debugpy
    DEBUG_CMD="python -m debugpy --listen 0.0.0.0:5678 -m"
fi

export CREDENTIALS_PATH=${CREDENTIALS_PATH:-$HOME/data/creds/credentials.json}
export NODE_PRIVATE_KEY=$(python $APPDIR/grid/bootstrap.py --private_key)
export NODE_UID=$(python $APPDIR/grid/bootstrap.py --uid)
export NODE_TYPE=$NODE_TYPE

echo "NODE_UID=$NODE_UID"
echo "NODE_TYPE=$NODE_TYPE"

# Check for IPv4 and IPv6 availability
IPV4_AVAILABLE=$(grep -c -v ":" /proc/net/if_inet6)
IPV6_AVAILABLE=$(grep -c ":" /proc/net/if_inet6)

# Default to IPv4 if no preference is set
HOST_MODE=${HOST_MODE:-ipv4}

if [[ $HOST_MODE == "ipv4" ]] && [[ $IPV4_AVAILABLE -gt 0 ]]; then
    # IPv4 is available
    HOST=${HOST:-0.0.0.0}
    exec $DEBUG_CMD hypercorn $RELOAD --bind $HOST:$PORT --log-level $LOG_LEVEL "$APP_MODULE"
elif [[ $HOST_MODE == "ipv6" ]] && [[ $IPV6_AVAILABLE -gt 0 ]]; then
    # IPv6 is available
    HOST6=${HOST6:-[::]}
    exec $DEBUG_CMD hypercorn $RELOAD --bind $HOST6:$PORT --log-level $LOG_LEVEL "$APP_MODULE"
elif [[ $HOST_MODE == "ipv4+ipv6" ]] && [[ $IPV4_AVAILABLE -gt 0 ]] && [[ $IPV6_AVAILABLE -gt 0 ]]; then
    # Both IPv4 and IPv6 are available
    HOST=${HOST:-0.0.0.0}
    HOST6=${HOST6:-[::]}
    exec $DEBUG_CMD hypercorn $RELOAD --bind $HOST:$PORT --bind $HOST6:$PORT --log-level $LOG_LEVEL "$APP_MODULE"
else
    echo "No suitable IP version available for the specified HOST_MODE: $HOST_MODE"
    exit 1
fi
