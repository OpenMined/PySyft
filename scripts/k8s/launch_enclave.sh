#!/bin/bash

# Enclave Node
bash -c '\
    export CLUSTER_NAME=testenclave1 CLUSTER_HTTP_PORT=9083 DEVSPACE_PROFILE=enclave && \
    tox -e dev.k8s.start -- --volume /sys/kernel/security:/sys/kernel/security --volume /dev/tmprm0:/dev/tmprm0 && \
    tox -e dev.k8s.hotreload'