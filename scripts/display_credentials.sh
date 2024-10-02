#!/bin/bash

# Construct variable names based on the cluster name
EMAIL_VAR="SYFT_LOGIN_${CLUSTER_NAME//[^a-zA-Z0-9_]/_}_ROOT_EMAIL"
PWD_VAR="SYFT_LOGIN_${CLUSTER_NAME//[^a-zA-Z0-9_]/_}_PASSWORD"

# Default CLIENT_NAME is "client"
CLIENT_NAME="client"

# Determine CLIENT_NAME based on CLUSTER_NAME
if [[ "$CLUSTER_NAME" == *"high"* ]]; then
    CLIENT_NAME="high_client"
elif [[ "$CLUSTER_NAME" == *"low"* ]]; then
    CLIENT_NAME="low_client"
fi

# Retrieve values from the constructed variable names
CLIENT_EMAIL="${!EMAIL_VAR}"
CLIENT_PWD="${!PWD_VAR}"

# Check if CLIENT_EMAIL or CLIENT_PWD are empty and provide a warning if needed
if [[ -z "$CLIENT_EMAIL" || -z "$CLIENT_PWD" ]]; then
    echo "Warning: CLIENT_EMAIL or CLIENT_PWD is empty. Please check the environment variables."
fi

# Output the formatted command
echo "\
To login to the Syft backend, copy and run the following command in Jupyter:

import syft as sy
$CLIENT_NAME = sy.login(
    email=\"${CLIENT_EMAIL}\",
    password=\"${CLIENT_PWD}\",
    port=\"${CLUSTER_HTTP_PORT}\"
)"
