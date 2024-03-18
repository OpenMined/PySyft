#!/bin/bash

# Gateway Node
bash -c '\
    export CLUSTER_NAME=testgateway1 CLUSTER_HTTP_PORT=9081 DEVSPACE_PROFILE=gateway && \
    tox -e dev.k8s.start && \
    tox -e dev.k8s.hotreload'