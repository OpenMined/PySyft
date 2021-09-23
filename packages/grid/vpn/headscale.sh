#!/bin/sh

if [ ! -f private.key ]
then
    wg genkey > private.key
fi

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=headscale
export NETWORK_NAME="${1}"
flask run -p 4000 --host=0.0.0.0&
headscale namespaces create $NETWORK_NAME || true
headscale serve
