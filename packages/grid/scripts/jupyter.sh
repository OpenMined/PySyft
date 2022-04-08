#!/bin/bash

# $1 is the PySyft dir
# $2 is the permission user like: om
# $3 is jupyter token

# this runs jupyter notebooks on port 8888
pidof -o %PPID -x $0 >/dev/null && echo "ERROR: Script $0 already running" && exit 1

echo "Starting Jupyter Server"

JUPYTER_CMD="cd ${1} && tox -e syft.jupyter ${3}"

/usr/sbin/runuser -l ${2} -c "$JUPYTER_CMD"
