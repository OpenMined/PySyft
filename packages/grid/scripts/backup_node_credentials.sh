#!/bin/bash

ROOT_DATA_PATH="$HOME/.syft/data"

docker_cp() {
    # Exit in case of error
    set -e

    # Define the container name pattern
    CONTAINER_NAME_PATTERN="backend-1"

    # Get a list of container names that match the pattern
    CONTAINER_NAMES=$(docker ps --filter "name=$CONTAINER_NAME_PATTERN" --format "{{.Names}}")

    if [ -z "$CONTAINER_NAMES" ]; then
        echo "No containers found with the name pattern: $CONTAINER_NAME_PATTERN"
        exit 1
    fi

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

    # Find all pods and namespaces with the specified image tag
    PODS_AND_NAMESPACES=($(kubectl get pods --all-namespaces -o=jsonpath="{range .items[*]}{.metadata.name}{'\t'}{.metadata.namespace}{'\t'}{range .spec.containers[*]}{.image}{'\n'}{end}{end}" | grep "$IMAGE_TAG" | awk '{print $1, $2}'))

    if [ ${#PODS_AND_NAMESPACES[@]} -eq 0 ]; then
        echo "No pods found with image tag: $IMAGE_TAG"
        exit 1
    fi

    for ((i = 0; i < ${#PODS_AND_NAMESPACES[@]}; i += 2)); do
        POD_NAME="${PODS_AND_NAMESPACES[i]}"
        NAMESPACE="${PODS_AND_NAMESPACES[i + 1]}"

        mkdir -p $ROOT_DATA_PATH/$NAMESPACE

        # Copy the file (suppress error message from kubectl cp command: "tar: Removing leading `/' from member names")
        kubectl cp "$NAMESPACE/$POD_NAME:$FILE_PATH_IN_POD" "$ROOT_DATA_PATH/$NAMESPACE/credentials.json" &>/dev/null

        # Check if the copy was successful
        if [ $? -eq 0 ]; then
            echo "Copied credentials.json from $POD_NAME in namespace $NAMESPACE to $ROOT_DATA_PATH/$NAMESPACE"
        else
            echo "Failed to copy credentials.json from $POD_NAME in namespace $NAMESPACE"
        fi
    done
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
    echo "d for Docker"
    echo "k for Kubernetes"
    read -p "Enter your choice: " choice

    case $choice in
        d)
            docker_cp
            ;;
        k)
            k8s_cp
            ;;
        *)
            echo "Invalid choice. Please select either 'd' for Docker or 'k' for Kubernetes."
            ;;
    esac
fi