#!/bin/sh

# Better to not run the worker server without ID
if [ -z "${WORKER_ID}" ]; then
  echo "[-] You need to set the WORKER_ID environment variable" 1&>2
  exit
fi

python worker-server.py --id $WORKER_ID --port 8777
