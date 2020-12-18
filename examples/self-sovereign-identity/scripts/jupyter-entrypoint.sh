#!/bin/sh

# Use the passed WORKSPACE DIRECTORY
# Otherwise use /workspace
if [ -z "${WORKSPACE_DIR}" ]; then
  WORKSPACE="/workspace"
else
  WORKSPACE="${WORKSPACE_DIR}"
fi

cd $WORKSPACE
echo "HELLO THIS IS PORT $1"
jupyter notebook --ip=0.0.0.0 --port="$1" --allow-root
