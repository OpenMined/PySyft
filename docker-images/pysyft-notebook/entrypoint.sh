#!/bin/sh

# Use the passed WORKSPACE DIRECTORY
# Otherwise use /workspace
if [ -z "${WORKSPACE_DIR}" ]; then
  WORKSPACE="/workspace"
else
  WORKSPACE="${WORKSPACE_DIR}"
fi

cd $WORKSPACE
jupyter notebook --ip=`cat /etc/hosts |tail -n 1|cut -f 1` --allow-root
