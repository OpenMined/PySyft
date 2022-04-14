#!/bin/ash

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=headscale
export NETWORK_NAME="${1}"
flask run -p 4000 --host=0.0.0.0&

# start server in background
headscale serve&

# Wait for headscale to start
sleep 10

# create namespace
headscale namespaces create $NETWORK_NAME || true

# kill background process
pgrep headscale | xargs kill -9

# start in foreground
headscale serve
