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

# Check if the file does not exist
if [ ! -f ~/.kube/config ]; then
    # Create a new kubeconfig file
    CONFIG_TEMPLATE="apiVersion: v1
clusters:
  - cluster:
      certificate-authority-data: <ca-placeholder>
      server: <server-placeholder>
    name: <cluster-name-placeholder>
contexts:
  - context:
      cluster: <cluster-name-placeholder>
      user: <user-placeholder>
    name: <cluster-name-placeholder>
current-context: <cluster-name-placeholder>
kind: Config
preferences: {}
users:
  - name: <user-placeholder>
    user:
      client-certificate-data: <cc-placeholder>
      client-key-data: <ck-placeholder>
      token: <token-placeholder>"

    # Replace placeholders with actual values
    echo "$CONFIG_TEMPLATE" | sed -e "s|<ca-placeholder>|$CERTIFICATE_AUTHORITY_DATA|g" \
        -e "s|<server-placeholder>|$SERVER|g" \
        -e "s|<cluster-name-placeholder>|$CLUSTER_NAME|g" \
        -e "s|<user-placeholder>|$USER_NAME|g" \
        -e "s|<cc-placeholder>|$CLIENT_CERTIFICATE_DATA|g" \
        -e "s|<ck-placeholder>|$CLIENT_KEY_DATA|g" \
        -e "s|<token-placeholder>|$TOKEN|g" \
        >> ~/.kube/config
    echo "Config File Created."
else
    # Append cluster Information to kubeconfig
    CLUSTER_UPDATED_JSON=$(jq -n \
        --arg ca_data "$CERTIFICATE_AUTHORITY_DATA" \
        --arg srv "$SERVER" \
        --arg cl_name "$CLUSTER_NAME" \
        '{
            "cluster": {
                "certificate-authority-data": $ca_data,
                "server": $srv
            },
            "name": $cl_name
        }')
    yq eval -i ".clusters += [$CLUSTER_UPDATED_JSON]" ~/.kube/config


    # Append context information to kubeconfig
    CONTEXT_UPDATED_JSON=$(jq -n \
        --arg cl_name "$CLUSTER_NAME" \
        --arg usr "$USER_NAME" \
        '{
            "context": {
                    "cluster": $cl_name,
                    "user": $usr
            },
            "name": $cl_name
        }')
    yq eval -i ".contexts += [$CONTEXT_UPDATED_JSON]" ~/.kube/config


    # Append user information to kubeconfig
    USER_UPDATED_JSON=$(jq -n \
        --arg cc_data "$CLIENT_CERTIFICATE_DATA" \
        --arg ck_data "$CLIENT_KEY_DATA" \
        --arg tkn "$TOKEN" \
        --arg usr "$USER_NAME" \
        '{
            "name": $usr,
            "user":{
                    "client-certificate-data": $cc_data,
                    "client-key-data": $ck_data,
                    "token": $tkn
            }
        }')
    yq eval -i ".users += [$USER_UPDATED_JSON]" ~/.kube/config

    yq eval -i ".current-context = \"$CLUSTER_NAME\" " ~/.kube/config
fi


echo "Updated kubeconfig with new cluster, context, and user"

