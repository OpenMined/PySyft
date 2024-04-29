#!/bin/sh

mkdir -p $SWFS_MASTER_DIR $SWFS_VOLUME_DIR $SWFS_VOLUME_IDX_DIR

# generate s3 config
# if s3config.json does not exist, then use template to generate one
if [ ! -f /run/secrets/seaweedfs/s3config.json ]; then
    echo "Generating s3 config"

    mkdir -m 600 -p /run/secrets/seaweedfs

    S3_ROOT_USER=${S3_ROOT_USER:-"admin"} \
    S3_ROOT_PWD=${S3_ROOT_PWD-$(openssl rand -hex 32)} \
    envsubst < /etc/secrets/s3config.template.json > /run/secrets/seaweedfs/s3config.json
fi

# setup cron jobs
echo "Setting up cron jobs"
cat > /etc/periodic/hourly/uncache <<EOF
#!/bin/sh
sh /root/swfs/scripts/wait_for_swfs.sh && echo remote.uncache -minAge=$SWFS_UNCACHE_MINAGE | weed shell
EOF

# start all processes
supervisord
