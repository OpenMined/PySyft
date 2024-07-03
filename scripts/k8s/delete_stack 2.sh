#!/bin/bash

# Deleting gateway node
bash -c "CLUSTER_NAME=testgateway1 tox -e dev.k8s.destroy || true"

# Deleting domain node
bash -c "CLUSTER_NAME=testdomain1 tox -e dev.k8s.destroy || true"