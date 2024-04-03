#!/bin/sh

mkdir -p $SWFS_MASTER_DIR $SWFS_VOLUME_DIR $SWFS_VOLUME_IDX_DIR

# generate s3 config
python -m src.cli.s3config --config=$S3_CONFIG_PATH --username=$S3_ROOT_USER --password=$S3_ROOT_PWD

# start all processes
supervisord
