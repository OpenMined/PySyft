#!/bin/bash

# Ensure CLUSTER_NAME is set
if [ -z "$CLUSTER_NAME" ]; then
  echo "CLUSTER_NAME is not set. Please set it before running the script."
  exit 1
fi

# Get the password from the secret and decode it
SYFT_PASSWORD=$(kubectl --context=k3d-$CLUSTER_NAME get secret backend-secret -n syft \
    -o jsonpath='{.data.defaultRootPassword}' | base64 --decode)

# Check if the command was successful
if [ $? -ne 0 ]; then
  echo "Failed to retrieve or decode the secret from the cluster."
  exit 1
fi


# Get the name of the backend pod (assuming there's only one, or picking the first one)
BACKEND_POD=$(kubectl --context=k3d-$CLUSTER_NAME get pods -n syft -l app.kubernetes.io/component=backend -o jsonpath='{.items[0].metadata.name}')

# Check if we successfully retrieved the pod name
if [ -z "$BACKEND_POD" ]; then
  echo "Failed to find the backend pod."
  exit 1
fi

# Get the root email from the environment variables of the backend pod
SYFT_ROOT_EMAIL=$(kubectl --context=k3d-$CLUSTER_NAME exec "$BACKEND_POD" -n syft \
    -- printenv DEFAULT_ROOT_EMAIL)

# Check if the command was successful
if [ $? -ne 0 ] || [ -z "$SYFT_ROOT_EMAIL" ]; then
  echo "Failed to retrieve the root email from the backend pod."
  exit 1
fi



# Export the root email as an environment variable
export SYFT_LOGIN_${CLUSTER_NAME//[^a-zA-Z0-9_]/_}_ROOT_EMAIL="$SYFT_ROOT_EMAIL"

# Export the password as an environment variable
export SYFT_LOGIN_${CLUSTER_NAME//[^a-zA-Z0-9_]/_}_PASSWORD="$SYFT_PASSWORD"


echo "Credentials successfully exported as environment variables."
echo "SYFT_LOGIN_${CLUSTER_NAME//[^a-zA-Z0-9_]/_}_ROOT_EMAIL=${SYFT_ROOT_EMAIL}"
echo "SYFT_LOGIN_${CLUSTER_NAME//[^a-zA-Z0-9_]/_}_PASSWORD=${SYFT_PASSWORD}"
