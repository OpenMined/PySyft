#!/bin/bash

# Function to check if a command exists
not_installed() {
    ! type "$1" &> /dev/null
}

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

# The string to match with context.user
USERNAME="$1"

# Ensure script is called with one argument
if [ -z "$1" ]; then
    echo "No username provided. Please provide it."
    echo "eg. ./load_aks_credentials.sh <username>"
    exit 1
fi

# Fetch data from 1Password
jsonData=$(op item get $USERNAME  --fields label=CLIENT_CERTIFICATE_DATA,label=USER_NAME,label=CLIENT_KEY_DATA,label=TOKEN,label=SERVER,label=CLUSTER_NAME,label=CERTIFICATE_AUTHORITY_DATA --format json)

# Check exit status of the last command
if [ $? -ne 0 ]; then
    exit 1
fi

# Assign the JSON values to variables
CLIENT_CERTIFICATE_DATA=$(echo $jsonData | jq -r '.[] | select(.label=="CLIENT_CERTIFICATE_DATA") | .value')
CLIENT_KEY_DATA=$(echo $jsonData | jq -r '.[] | select(.label=="CLIENT_KEY_DATA") | .value')
TOKEN=$(echo $jsonData | jq -r '.[] | select(.label=="TOKEN") | .value')
SERVER=$(echo $jsonData | jq -r '.[] | select(.label=="SERVER") | .value')
CLUSTER_NAME=$(echo $jsonData | jq -r '.[] | select(.label=="CLUSTER_NAME") | .value')
CERTIFICATE_AUTHORITY_DATA=$(echo $jsonData | jq -r '.[] | select(.label=="CERTIFICATE_AUTHORITY_DATA") | .value')
USER_NAME=$(echo $jsonData | jq -r '.[] | select(.label=="USER_NAME") | .value')

kubectl config set-cluster "$CLUSTER_NAME" --server="$SERVER"
kubectl config set-credentials "$USER_NAME" --token="$TOKEN"
kubectl config set-context "$CLUSTER_NAME" --cluster="$CLUSTER_NAME" --user="$USER_NAME" --namespace=default
kubectl config use-context "$CLUSTER_NAME"
kubectl config set clusters."$CLUSTER_NAME".certificate-authority-data "$CERTIFICATE_AUTHORITY_DATA"
kubectl config set users."$USER_NAME".client-certificate-data "$CLIENT_CERTIFICATE_DATA"
kubectl config set users."$USER_NAME".client-key-data "$CLIENT_KEY_DATA"


echo "Updated kubeconfig with new cluster, context, and user"

