#!/bin/bash

# The string to match with context.name in the kubeconfig
CONTEXT_NAME="$1"

# File path to kubeconfig
KUBECONFIG_FILE="~/.kube/config"

# Parse and find user and cluster details from the kubectl config
parse_kubectl_config() {
    # Finding the context with the matching user
    USERNAME=$(yq e ".contexts[] | select(.name == \"$CONTEXT_NAME\") | .context.user" ~/.kube/config)
    CLUSTER_NAME=$(yq e ".contexts[] | select(.name == \"$CONTEXT_NAME\") | .context.cluster" ~/.kube/config)

    # Check if the context was found
    if [ -z "$USERNAME" ]; then
       echo "No matching context found."
       exit 1
    fi

    # Extracting user details
    CLIENT_CERTIFICATE_DATA=$(yq e ".users[] | select(.name == \"$USERNAME\") | .user.\"client-certificate-data\"" ~/.kube/config)
    CLIENT_KEY_DATA=$(yq e ".users[] | select(.name == \"$USERNAME\") | .user.\"client-key-data\"" ~/.kube/config)
    TOKEN=$(yq e ".users[] | select(.name == \"$USERNAME\") | .user.token" ~/.kube/config)

    # Extracting cluster details
    CERTIFICATE_AUTHORITY_DATA=$(yq e ".clusters[] | select(.name == \"$CLUSTER_NAME\") | .cluster.\"certificate-authority-data\"" ~/.kube/config)
    SERVER=$(yq e ".clusters[] | select(.name == \"$CLUSTER_NAME\") | .cluster.server" ~/.kube/config)

    # Creating a new item in 1Password
    op item create --category="Api Credential" --title="$CONTEXT_NAME" CLIENT_CERTIFICATE_DATA=$CLIENT_CERTIFICATE_DATA USER_NAME=$USERNAME CLIENT_KEY_DATA=$CLIENT_KEY_DATA TOKEN=$TOKEN CERTIFICATE_AUTHORITY_DATA=$CERTIFICATE_AUTHORITY_DATA SERVER=$SERVER CLUSTER_NAME=$CLUSTER_NAME

    # Check exit status of the op item creation command
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

# Ensure script is called with two arguments
if [ -z "$1" ]; then
    echo "Insufficient arguments provided. Please provide context name."
    echo "eg. ./save_aks_credentials.sh <context_name>"
    exit 1
fi

# Function to check if a command exists
not_installed() {
    ! type "$1" &> /dev/null
}

# Define associative array for command installation links
declare -A INSTALLATION_LINKS=(
    [jq]="https://stedolan.github.io/jq/download/"
    [yq]="https://github.com/mikefarah/yq"
    [op]="https://developer.1password.com/docs/cli/get-started/"
)

# Check for required commands and provide installation links
for cmd in jq yq op; do
    if not_installed "$cmd"; then
        echo "$cmd is not installed. You can install it from: ${INSTALLATION_LINKS[$cmd]}"
        exit 1
    fi
done

# Invoke the parsing function
parse_kubectl_config
