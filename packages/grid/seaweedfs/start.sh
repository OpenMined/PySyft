#! /usr/bin/env bash

sleep 30 &&
echo "s3.configure -access_key admin -secret_key admin -user iam -actions Read,Write,List,Tagging,Admin -apply" \
| weed shell > /dev/null 2>&1 \
& weed server -s3 -s3.port=8333 -master.volumeSizeLimitMB=1024