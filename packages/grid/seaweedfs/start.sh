#! /usr/bin/env bash

sleep 30 &&
echo "s3.configure -access_key ${S3_ROOT_USER} -secret_key ${S3_ROOT_PWD} -user iam -actions Read,Write,List,Tagging,Admin -apply" \
| weed shell > /dev/null 2>&1 \
& weed server -s3 -s3.port=${S3_PORT} -master.volumeSizeLimitMB=${S3_VOLUME_SIZE_MB}