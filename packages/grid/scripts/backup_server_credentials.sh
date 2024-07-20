#!/bin/bash

#Define destination path on the host machine
ROOT_DATA_PATH="$HOME/.syft/data"

# Define the source path of the credentials.json file in the container
SOURCE_PATH="/root/data/creds/credentials.json"


calculate_checksum() {
    # Calculate the checksum of a file
    local file="$1"
    local checksum
    checksum=$(sha256sum "$file" | awk '{print $1}')
    echo "$checksum"
}

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

    for CONTAINER_NAME in $CONTAINER_NAMES; do


        # Define the destination path on the host machine
        DESTINATION_PATH="$ROOT_DATA_PATH/$CONTAINER_NAME"

        # Calculate the checksum of the source file before copying
        SOURCE_CHECKSUM=$(docker exec "$CONTAINER_NAME" sha256sum "$SOURCE_PATH" | awk '{print $1}')

        # Check if the destination directory already exists
        if [ ! -d "$DESTINATION_PATH" ]; then
            mkdir -p "$DESTINATION_PATH"
        fi

        # Copy the credentials.json file from the container to the host, placing it in the destination folder
        # Suppress output from docker cp command.
        docker cp "${CONTAINER_NAME}:${SOURCE_PATH}" "$DESTINATION_PATH/credentials.json" >/dev/null

        # Check if the copy was successful
        if [ $? -eq 0 ]; then
            # Calculate the checksum of the destination file
            DESTINATION_CHECKSUM=$(calculate_checksum "$DESTINATION_PATH/credentials.json")

            # Check if the source and destination checksums match
            if [ "$SOURCE_CHECKSUM" = "$DESTINATION_CHECKSUM" ]; then
                echo "Copied credentials.json from container $CONTAINER_NAME to $DESTINATION_PATH/credentials.json"
            else
                echo "Failed to copy credentials.json from $CONTAINER_NAME. Checksum mismatch."
            fi
        else
            echo "Failed to copy credentials.json from $CONTAINER_NAME"
        fi
    done
}


k8s_cp() {
    IMAGE_TAG="syft-backend"

    # Get a list of available contexts
    CONTEXTS=($(kubectl config get-contexts -o name))

    for CONTEXT in "${CONTEXTS[@]}"; do
        # Skip the "docker-desktop" context
        if [ "$CONTEXT" = "docker-desktop" ]; then
            continue
        fi

        # Set the context for kubectl
        kubectl config use-context "$CONTEXT"

        # Find all pods and namespaces with the specified image tag
        PODS_AND_NAMESPACES=($(kubectl get pods --all-namespaces -o=jsonpath="{range .items[*]}{.metadata.name}{'\t'}{.metadata.namespace}{'\t'}{range .spec.containers[*]}{.image}{'\n'}{end}{end}" | grep "$IMAGE_TAG" | awk '{print $1, $2}'))

        if [ ${#PODS_AND_NAMESPACES[@]} -eq 0 ]; then
            echo "No pods found with image tag: $IMAGE_TAG in context $CONTEXT"
        else
            for ((i = 0; i < ${#PODS_AND_NAMESPACES[@]}; i += 2)); do
                POD_NAME="${PODS_AND_NAMESPACES[i]}"
                NAMESPACE="${PODS_AND_NAMESPACES[i + 1]}"

                DESTINATION_FOLDER="$ROOT_DATA_PATH/$CONTEXT""_""$NAMESPACE"
                mkdir -p $DESTINATION_FOLDER

                # Calculate the checksum of the source file inside the pod
                SOURCE_CHECKSUM=$(kubectl exec -n "$NAMESPACE" -it "$POD_NAME" -- sha256sum "$SOURCE_PATH" | awk '{print $1}')

                # Copy the file (suppress error message from kubectl cp command: "tar: Removing leading `/' from member names")
                kubectl cp "$NAMESPACE/$POD_NAME:$SOURCE_PATH" "$DESTINATION_FOLDER/credentials.json" &>/dev/null

                # Check if the copy was successful
                if [ $? -eq 0 ]; then
                    # Calculate the checksum of the destination file
                    DESTINATION_CHECKSUM=$(calculate_checksum "$DESTINATION_FOLDER/credentials.json")

                    # Check if the checksums match
                    if [ "$SOURCE_CHECKSUM" = "$DESTINATION_CHECKSUM" ]; then
                        echo "Copied credentials.json from $POD_NAME in namespace $NAMESPACE in context $CONTEXT to $DESTINATION_FOLDER"
                    else
                        echo "Failed to copy credentials.json. Checksum mismatch."
                    fi
                else
                    echo "Failed to copy credentials.json from $POD_NAME in namespace $NAMESPACE in context $CONTEXT."
                fi
            done
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