# ---------------------------------------------------------------------------------------------------------------------

[group('cloud')]
[private]
check-platform:
    #!/bin/bash
    set -euo pipefail

    OSTYPE=$(uname -sm)
    MSG="==================================================================================================\n\
    Deploying dev->cloud k8s (x64 nodes) requires images to be built with --platform=linux/amd64\n\
    On Apple Silicon, cross-platform image is unstable on different providers\n\n\
    Current status:\n\
    ✅ | Docker Desktop | 4.34.0+ | *Enable* containerd and *uncheck* 'Use Rosetta for x86_64/amd64...'\n\
    ❌ | OrbStack       | 1.7.2   | Rosetta: gets stuck & qemu: errors with 'illegal instruction'\n\
    ❌ | Lima VM/Colima | 0.23.2  | Rosetta: gets stuck & qemu: errors with 'illegal instruction'\n\
    =================================================================================================="

    if [[ "$OSTYPE" == "Darwin arm64" ]]; then
        echo -e $MSG
    fi

[group('cloud')]
[private]
deploy-cloud cluster_ctx registry_url namespace profile: check-platform
    #!/bin/bash

    CONTEXT_NAME=$(kubectl config get-contexts -o=name | grep "{{ cluster_ctx }}")

    if [ -z "$CONTEXT_NAME" ]; then
        echo "Context not found: {{ cluster_ctx }}. Authorized with cloud providers to get relevant K8s cluster contexts"
        exit 1
    fi

    set -euo pipefail

    # cloud deployments always have tracing false + platform=amd64
    just tracing=false registry_url={{ registry_url }} \
        deploy-devspace $CONTEXT_NAME {{ namespace }} "-p {{ profile }} --var PLATFORM=amd64"

[group('cloud')]
[private]
purge-cloud cluster_ctx namespace:
    #!/bin/bash

    CONTEXT_NAME=$(kubectl config get-contexts -o=name | grep "{{ cluster_ctx }}")

    if [ -z "$CONTEXT_NAME" ]; then
        echo "Context not found: {{ cluster_ctx }}. Authorized with cloud providers to get relevant K8s cluster contexts"
        exit 1
    fi

    set -euo pipefail

    just purge-devspace $CONTEXT_NAME {{ namespace }}
    kubectl delete ns {{ namespace }} --force --grace-period=0 --context $CONTEXT_NAME

# ---------------------------------------------------------------------------------------------------------------------

# Auth all components required for deploying Syft to Google Cloud
[group('cloud-gcp')]
auth-gcloud:
    #!/bin/bash
    set -euo pipefail

    # login to gcloud
    ACCOUNT=$(gcloud config get-value account)
    if [ -z "$ACCOUNT" ]; then
        gcloud auth login
    fi

    echo "Logged in as \"$(gcloud config get-value account)\""

    # install gke-gcloud-auth-plugin
    gke_installed=$(gcloud components list --only-local-state --filter gke-gcloud-auth-plugin --format=list 2>/dev/null)
    if [ -z "$gke_installed" ]; then
        gcloud components install gke-gcloud-auth-plugin
        echo "Installed gke-gcloud-auth-plugin"
    fi

# Deploy local code as datasite-high to Google Kubernetes Engine
[group('cloud-gcp')]
deploy-gcp-high gcp_cluster gcp_registry_url namespace="syft": (deploy-cloud gcp_cluster gcp_registry_url namespace "gcp")

# Deploy local code as datasite-high to Google Kubernetes Engine
[group('cloud-gcp')]
deploy-gcp-low gcp_cluster gcp_registry_url namespace="syft": (deploy-cloud gcp_cluster gcp_registry_url namespace "gcp-low")

# Purge deployment from a cluster
[group('cloud-gcp')]
purge-gcp gcp_cluster namespace="syft": (purge-cloud gcp_cluster namespace)

# ---------------------------------------------------------------------------------------------------------------------

[group('cloud-az')]
auth-az tenant="creditsopenmined.onmicrosoft.com":
    #!/bin/bash

    # login to azure
    ACCOUNT=$(az account show --query user.name)
    if [ -z "$ACCOUNT" ]; then
        az login --tenant {{ tenant }}
    fi

    echo "Logged in as $(az account show --query user.name)"

# Deploy local code as datasite-high to Azure Kubernetes Service
[group('cloud-az')]
deploy-az-high aks_cluster az_registry namespace="syft": (deploy-cloud aks_cluster az_registry namespace "azure")

