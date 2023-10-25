#!/bin/bash

ROOT_DATA_PATH="$HOME/.syft/data"

docker_cp() {
    # Exit in case of error
    set -e

    # Define the container name pattern
    CONTAINER_NAME_PATTERN="backend-1"

    # Get a list of container names that match the pattern
    CONTAINER_NAMES=$(docker ps --filter "name=$CONTAINER_NAME_PATTERN" --format "{{.Names}}")

    mkdir -p "$ROOT_DATA_PATH"

    # Loop through each matching container
    for CONTAINER_NAME in $CONTAINER_NAMES; do
        # Define the source path of the credentials.json file in the container
        SOURCE_PATH="/storage/credentials.json"

        # Define the destination path on the host machine
        DESTINATION_PATH="$ROOT_DATA_PATH/$CONTAINER_NAME"

        # Create the directory for the specific container
        mkdir -p "$DESTINATION_PATH"

        # Copy the credentials.json file from the container to the host
        docker cp "${CONTAINER_NAME}:${SOURCE_PATH}" "$DESTINATION_PATH"

        # Check if the copy was successful
        if [ $? -eq 0 ]; then
            echo "Copied credentials.json from $CONTAINER_NAME to $DESTINATION_PATH"
        else
            echo "Failed to copy credentials.json from $CONTAINER_NAME"
        fi
    done
}


# Not tested for multiple pods
k8s_cp() {
    IMAGE_TAG="grid-backend"
    FILE_PATH_IN_POD="/storage/credentials.json"

    # Find the pod and namespace
    read -r POD_NAME NAMESPACE <<< $(kubectl get pods --all-namespaces -o=jsonpath="{range .items[*]}{.metadata.name}{'\t'}{.metadata.namespace}{'\t'}{range .spec.containers[*]}{.image}{'\n'}{end}{end}" | grep "$IMAGE_TAG" | awk '{print $1, $2}')

    # Check if we found the pod and namespace
    if [ -z "$POD_NAME" ] || [ -z "$NAMESPACE" ]; then
        echo "No pod found with image tag: $IMAGE_TAG"
        exit 1
    fi

    mkdir -p $ROOT_DATA_PATH/$NAMESPACE

    # Copy the file
    kubectl cp "$NAMESPACE/$POD_NAME:$FILE_PATH_IN_POD" $ROOT_DATA_PATH/$NAMESPACE/credentials.json
}


# Check if the "--docker" flag is set
if [[ "$1" == "--docker" ]]
then
    docker_cp

# Check if the "--k8s" flag is set
elif [[ "$1" == "--k8s" ]]
then
    k8s_cp

# If no flag is set, prompt the user for their choice
else
    echo "Please select an option:"
    echo "1. Docker"
    echo "2. Kubernetes"
    read -p "Enter the number of your choice: " choice

    case $choice in
        1)
            docker_cp
            ;;
        2)
            k8s_cp
            ;;
        *)
            echo "Invalid choice. Please select either 1 or 2."
            ;;
    esac
fi

