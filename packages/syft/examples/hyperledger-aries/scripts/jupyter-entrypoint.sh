#!/bin/bash

# Use the passed WORKSPACE DIRECTORY
# Otherwise use /workspace
if [ -z "${WORKSPACE_DIR}" ]; then
    WORKSPACE="/workspace"
else
    WORKSPACE="${WORKSPACE_DIR}"
fi

cd "$WORKSPACE" || exit
echo "Running Jupyter on PORT=$1"
jupyter lab --ip=0.0.0.0 --port="$1" --allow-root
