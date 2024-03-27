#!/bin/bash

# generate s3 secret
python src/configure_s3.py

# start all processes
supervisord
