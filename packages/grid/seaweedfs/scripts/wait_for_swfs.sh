#!/bin/sh

set -e

# Retry until 2 minutes
RETRY_ARGS="--retry 24 --retry-delay 5 --retry-all-errors"
MASTER_URL="localhost:9333"
FILER_URL="localhost:8888"

curl --silent $RETRY_ARGS http://$MASTER_URL/cluster/healthz > /dev/null
curl --silent $RETRY_ARGS http://$MASTER_URL/vol/status > /dev/null
curl --silent $RETRY_ARGS http://$FILER_URL/ > /dev/null
