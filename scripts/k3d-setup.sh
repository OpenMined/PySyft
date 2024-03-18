#!/bin/bash

# Initialize default values
VERSION=""
CLUSTER_NAME="syft-test" # Default name for K3d cluster
NAMESPACE="syft" # Default namespace

# Function to display usage
usage() {
    echo "Usage: $0 --version <version> [--cluster-name <cluster_name>] [--namespace <namespace>]"
    exit 1
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Validate the version
if [[ -z "$VERSION" ]]; then
    echo "The --version option is required."
    usage
fi

# Check if the cluster already exists
if k3d cluster list | grep -qw "$CLUSTER_NAME"; then
    echo "Deleting existing K3d cluster named $CLUSTER_NAME"
    k3d cluster delete "$CLUSTER_NAME"
fi

# Create the K3d cluster
echo "Creating K3d cluster named $CLUSTER_NAME"
k3d cluster create "$CLUSTER_NAME" -p 8080:80@loadbalancer

# Setup Helm for Syft
echo "Setting up Helm for Syft"
helm repo add openmined https://openmined.github.io/PySyft/helm
helm repo update

# Provision Helm Charts for Syft
echo "Provisioning Helm Charts for Syft in namespace $NAMESPACE"
helm install my-domain openmined/syft --version "$VERSION" --namespace "$NAMESPACE" --create-namespace
