#!/bin/bash

# The string to match with context.user
MATCH_STRING="$1"

# File path to kubeconfig
KUBECONFIG_FILE="~/.kube/config"

# Parse and find user and cluster details from the kubectl config
parse_kubectl_config() {
    # Finding the context with the matching user
    CONTEXT_NAME=$(yq e ".contexts[] | select(.context.user == \"$MATCH_STRING\") | .name" ~/.kube/config)

    # Check if the context was found
    if [ -z "$CONTEXT_NAME" ]; then
       echo "No matching user found."
       exit 1
    fi

    # Extracting cluster name from the context
    CLUSTER_NAME=$(yq e ".contexts[] | select(.name == \"$CONTEXT_NAME\") | .context.cluster" ~/.kube/config)

    # Extracting user details
    CLIENT_CERTIFICATE_DATA=$(yq e ".users[] | select(.name == \"$MATCH_STRING\") | .user.\"client-certificate-data\"" ~/.kube/config)
    CLIENT_KEY_DATA=$(yq e ".users[] | select(.name == \"$MATCH_STRING\") | .user.\"client-key-data\"" ~/.kube/config)
    TOKEN=$(yq e ".users[] | select(.name == \"$MATCH_STRING\") | .user.token" ~/.kube/config)

    # Extracting cluster details
    CERTIFICATE_AUTHORITY_DATA=$(yq e ".clusters[] | select(.name == \"$CLUSTER_NAME\") | .cluster.\"certificate-authority-data\"" ~/.kube/config)
    SERVER=$(yq e ".clusters[] | select(.name == \"$CLUSTER_NAME\") | .cluster.server" ~/.kube/config)

    # Creating a new item in 1Password
    op item create --category="Api Credential" --title="$MATCH_STRING" CLIENT_CERTIFICATE_DATA=$CLIENT_CERTIFICATE_DATA USER_NAME=$MATCH_STRING CLIENT_KEY_DATA=$CLIENT_KEY_DATA TOKEN=$TOKEN CERTIFICATE_AUTHORITY_DATA=$CERTIFICATE_AUTHORITY_DATA SERVER=$SERVER CLUSTER_NAME=$CLUSTER_NAME

    # Check exit status of the op item creation command
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

# Ensure script is called with one argument
if [ -z "$1" ]; then
    echo "No username provided. Please provide it."
    echo "eg. ./save_aks_credentials.sh <username>"
    exit 1
fi

# Check for jq
if not_installed jq; then
    echo "jq is not installed. You can install it from: https://stedolan.github.io/jq/download/"
    exit 1
fi

# Check for yq
if not_installed yq; then
    echo "yq is not installed. You can install it from: https://github.com/mikefarah/yq"
    exit 1
fi

# Check for op
if not_installed op; then
    echo "op is not installed. You can install it from: https://developer.1password.com/docs/cli/get-started/"
    exit 1
fi


# Invoke the parsing function
parse_kubectl_config
