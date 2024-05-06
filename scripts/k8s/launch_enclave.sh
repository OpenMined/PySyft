#!/bin/bash

# Enclave Node
bash -c '\
    export CLUSTER_NAME=testenclave1 CLUSTER_HTTP_PORT=9083 DEVSPACE_PROFILE=enclave && \
    tox -e dev.k8s.start && \
    tox -e dev.k8s.hotreload'