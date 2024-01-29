#!/usr/bin/env bash

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=veilid
/veilid/veilid-server -c /veilid/veilid-server.conf --debug &
flask run -p 4000 --host=0.0.0.0

