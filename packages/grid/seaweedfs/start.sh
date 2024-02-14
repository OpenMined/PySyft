#!/usr/bin/env bash

S3_PORT=${S3_PORT:-8333}
S3_VOLUME_SIZE_MB=${S3_VOLUME_SIZE_MB:-1024}
S3_ROOT_USER=${S3_ROOT_USER:-admin}
S3_ROOT_PWD=${S3_ROOT_PWD:-password}
SEAWEED_MOUNT_PORT=${SEAWEED_MOUNT_PORT:-4001}

weed server -s3 -s3.port="$S3_PORT" -volume.max=500 -master.volumeSizeLimitMB="$S3_VOLUME_SIZE_MB" &
echo "s3.configure -access_key $S3_ROOT_USER -secret_key $S3_ROOT_PWD \
-user iam -actions Read,Write,List,Tagging,Admin -apply" | weed shell > /dev/null 2>&1

flask run -p "$SEAWEED_MOUNT_PORT" --host=0.0.0.0
