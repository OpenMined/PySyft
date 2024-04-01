#!/bin/sh

# generate s3 configu
python -m src.cli.s3_config

# mount provisioned buckets
python -m src.cli.automount

# start all processes
supervisord
