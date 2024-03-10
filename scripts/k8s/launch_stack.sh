#!/bin/bash

# Domain Node
bash -c '\
    export CLUSTER_NAME=testgateway1 CLUSTER_HTTP_PORT=9081 DEVSPACE_PROFILE=gateway && \
    tox -e dev.k8s.start && \
    tox -e dev.k8s.hotreload'

# Gateway Node
bash -c '\
    export CLUSTER_NAME=testdomain1 CLUSTER_HTTP_PORT=9082 && \
    tox -e dev.k8s.start && \
    tox -e dev.k8s.hotreload'