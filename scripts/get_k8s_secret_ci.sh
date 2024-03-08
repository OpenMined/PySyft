#!/bin/bash
export SYFT_LOGIN_testgateway1_PASSWORD=$(kubectl --context=k3d-testgateway1 get secret backend-secret -n syft \
    -o jsonpath='{.data.defaultRootPassword}' | base64 --decode)

export SYFT_LOGIN_testdomain1_PASSWORD=$(kubectl get --context=k3d-testdomain1 secret backend-secret -n syft \
    -o jsonpath='{.data.defaultRootPassword}' | base64 --decode)
