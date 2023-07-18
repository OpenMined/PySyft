#!/bin/bash

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=headscale
export NETWORK_NAME="${1}"

# start server in background
headscale serve&

# Wait for headscale to start
waitforit -address=http://localhost:8080/health -status=200 -timeout=60 -- echo "server started"

# create users
headscale users create $NETWORK_NAME || true

#
# headscale -u omnet preauthkeys create -o json

# kill background process
# pgrep headscale | xargs kill -9

# # start in foreground
# headscale serve
flask run -p 4000 --host=0.0.0.0