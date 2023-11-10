#!/bin/sh
echo "got api key"
echo ${STACK_API_KEY}
export STACK_API_KEY=${STACK_API_KEY}

echo "s3.configure -access_key $S3_ROOT_USER -secret_key $S3_ROOT_PWD \
-user iam -actions Read,Write,List,Tagging,Admin -apply" | weed shell > /dev/null 2>&1 &
weed server -s3 -s3.port=$S3_PORT -volume.max=500 -master.volumeSizeLimitMB=$S3_VOLUME_SIZE_MB &
flask run -p $SEAWEED_MOUNT_PORT --host=0.0.0.0
