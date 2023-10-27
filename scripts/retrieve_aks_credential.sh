#!/bin/bash

# Azure Cluster Credentials Retrieval Script

# Overview:
# This script retrieves Azure Kubernetes Service (AKS) cluster credentials using 'kubectl-passman' and sets them
# as a new Kubernetes configuration. It allows you to specify the username for which credentials should be retrieved.

# Prerequisites:
# 1. The 'kubectl passman' plugin must be installed to manage 1Password records.
# 2. The 'kubectl' command-line tool must be installed.
# 3. 'kubectl krew' should be installed.
# 4. '1Password CLI' (op) should be installed.
# 5. Ensure 'jq' (a command-line JSON processor) is available.

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed. Please install it before proceeding."
    echo "kubectl installation: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

# Check if 1password CLI is installed
if ! command -v op &> /dev/null; then
    echo "1password CLI is not installed. Please install it before proceeding."
    echo "Installation instructions: https://support.1password.com/command-line-getting-started/"
    exit 1
fi

# Check if 1password CLI is logged in
if ! op vault list &> /dev/null; then
    echo "1password CLI is not logged in. Please log in using 'op signin' before proceeding."
    exit 1
fi

# Check if kube krew is installed
if ! kubectl krew &> /dev/null; then
    echo "kube krew is not installed. Do you want to install it? (y/n)"
    read -r install_krew
    if [ "$install_krew" == "y" ]; then
        (
        set -x; cd "$(mktemp -d)" &&
        OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
        ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')" &&
        KREW="krew-${OS}_${ARCH}" &&
        curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW}.tar.gz" &&
        tar zxvf "${KREW}.tar.gz" &&
        ./"${KREW}" install krew
        )
        export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"
    else
        echo "kube krew is required for this script to work. Exiting."
        exit 1
    fi
fi

# Check if "kubectl passman" plugin is installed
if ! kubectl krew list | grep -q '^passman$'; then
    echo "kubectl passman plugin is not installed. Do you want to install it? (y/n)"
    read -r install_passman
    if [ "$install_passman" == "y" ]; then
        kubectl krew install passman
    else
        echo "kubectl passman plugin is required for this script to work. Exiting."
        exit 1
    fi
fi

# Prompt for the name
read -p "Enter the user name : " name

# Set the retrieved credentials as a new Kubernetes configuration
kubectl config set-credentials "$name" --exec-api-version=client.authentication.k8s.io/v1beta1  --exec-command=kubectl-passman --exec-arg=1password --exec-arg="$name"


# Updating the context
echo "Updating the kubeconfig context..."
kubectl config get-contexts

echo "Credentials have been set as a new Kubernetes configuration."
