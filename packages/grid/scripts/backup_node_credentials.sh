#!/usr/bin/env bash

# Exit in case of error
set -e

ROOT_DATA_PATH="$HOME/.syft/data"

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

