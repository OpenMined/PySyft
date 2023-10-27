#!/bin/bash

# Azure Cluster Credentials Storage Script for 1Password

# Overview:
# This script is designed to simplify the process of storing Azure Kubernetes Service (AKS) cluster credentials
# securely in a 1Password vault. It automates the retrieval of cluster credentials using kubectl and stores
# them in a 1Password record for future access.

# Prerequisites:
# 1. 1Password Command-Line Interface (CLI) must be installed. If not installed, follow the installation instructions:
#    https://support.1password.com/command-line-getting-started/
# 2. You should be logged in to 1Password using the 'op signin' command.
# 3. The 'kubectl' command-line tool and 'kubectl krew' plugin manager must be installed.
# 4. The 'kubectl passman' plugin must be installed to manage 1Password records.
# 5. Ensure 'jq' (a command-line JSON processor) is available.

# Usage:
# 1. Run az login to log in to Azure.
# 2. Run az aks get-credentials --resource-group <resource-group-name> --name <cluster-name> to get the cluster credentials.
# 3. Run this script, and it will guide you through the process of storing Azure cluster credentials in 1Password.
# 4. If the 1Password CLI, kubectl, or krew is not installed, the script will provide instructions for installation.
# 5. If the 'kubectl passman' plugin is not installed, it will ask if you want to install it.
# 6. Enter the username associated with your Azure cluster credentials when prompted.
# 7. The script will retrieve the cluster credentials using kubectl and store them securely in 1Password.
# 6. Once the credentials are stored, they can be easily accessed using 'kubectl passman' commands.

# Important Note:
# Make sure that you have appropriate permissions and access to 1Password, as well as the necessary credentials
# and access to the Azure Kubernetes Service cluster.

# Caution:
# Keep your 1Password master password and account credentials secure, as they are crucial for accessing
# and managing stored credentials.


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
if ! command -v kubectl krew &> /dev/null; then
    echo "kube krew is not installed. Do you want to install it? (y/n)"
    read -r install_krew
    if [ "$install_krew" == "y" ]; then
        (
            set -x
            cd "$(mktemp -d)" &&
            OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
            ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/armv7l/arm/' -e 's/aarch64/arm64/')" &&
            curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/krew.tar.gz" &&
            tar zxvf krew.tar.gz &&
            KREW=./krew-"$OS"_"$ARCH" &&
            "$KREW" install krew
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

# Run the kubectl and jq command with the provided name
config_token=$(kubectl config view --raw -o json | jq --arg name "$name" '.users[] | select(.name==$name) | .user' -c)

# Check if the config_token is empty
if [ -z "$config_token" ]; then
  echo "Error: No configuration token found for name '$name'"
  exit 1
fi

# Set the CONFIG_TOKEN environment variable
export CONFIG_TOKEN="$config_token"


# Execute the specified command with CONFIG_TOKEN as an argument
kubectl passman 1password "$name" "$CONFIG_TOKEN"
