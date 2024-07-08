#!/bin/bash

# Initialize default values
VERSION=""
NODE_NAME=""
NODE_SIDE_TYPE="high" # Default value for NODE_SIDE_TYPE
NODE_TYPE=""
PORT=""
DEFAULT_ROOT_EMAIL=""
DEFAULT_ROOT_PASSWORD=""

# Function to display usage
usage() {
    echo "Usage: $0 -v|--version <version> -n|--name <node_name> -t|--type <node_type> -p|--port <port> [-s|--side <node_side_type>] [--root-email <default_root_email>] [--root-password <default_root_password>]"
    exit 1
}

# Function to check if a port is occupied
is_port_occupied() {
    local port=$1
    if lsof -i:"$port" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to validate Docker version pattern
is_valid_version() {
    local version=$1
    if [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta|rc)\.[0-9]+)?$ ]]; then
        return 0
    else
        return 1
    fi
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -n|--name)
            NODE_NAME="$2"
            shift 2
            ;;
        -s|--side)
            NODE_SIDE_TYPE="$2"
            shift 2
            ;;
        -t|--type)
            NODE_TYPE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --root-email)
            DEFAULT_ROOT_EMAIL="$2"
            shift 2
            ;;
        --root-password)
            DEFAULT_ROOT_PASSWORD="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Debug output to verify parsed values
echo "VERSION: $VERSION"
echo "NODE_NAME: $NODE_NAME"
echo "NODE_SIDE_TYPE: $NODE_SIDE_TYPE"
echo "NODE_TYPE: $NODE_TYPE"
echo "PORT: $PORT"
echo "DEFAULT_ROOT_EMAIL: $DEFAULT_ROOT_EMAIL"
echo "DEFAULT_ROOT_PASSWORD: $DEFAULT_ROOT_PASSWORD"

# Check if all required options are set
if [[ -z "$VERSION" || -z "$NODE_NAME" || -z "$NODE_TYPE" || -z "$PORT" ]]; then
    echo "All options are required."
    usage
fi

# Validate Docker version pattern
if ! is_valid_version "$VERSION"; then
    echo "Invalid version format. Expected format: X.Y.Z or X.Y.Z-suffix.N (e.g., 0.8.2 or 0.8.2-beta.6)"
    exit 1
fi

# Check if the specified port is occupied
if is_port_occupied "$PORT"; then
    echo "Port $PORT is already in use. Please choose a different port."
    exit 1
fi

# Run the Podman command with the specified environment variables in detached mode
PODMAN_RUN_CMD="podman run --rm -d \
    --name \"$NODE_NAME\" \
    -e VERSION=\"$VERSION\" \
    -e NODE_NAME=\"$NODE_NAME\" \
    -e NODE_SIDE_TYPE=\"$NODE_SIDE_TYPE\" \
    -e NODE_TYPE=\"$NODE_TYPE\" \
    -e PORT=\"$PORT\" \
    -e SINGLE_CONTAINER_MODE=true \
    -p \"$PORT:$PORT\""

# Add optional environment variables if provided
if [[ -n "$DEFAULT_ROOT_EMAIL" ]]; then
    PODMAN_RUN_CMD="$PODMAN_RUN_CMD -e DEFAULT_ROOT_EMAIL=\"$DEFAULT_ROOT_EMAIL\""
fi

if [[ -n "$DEFAULT_ROOT_PASSWORD" ]]; then
    PODMAN_RUN_CMD="$PODMAN_RUN_CMD -e DEFAULT_ROOT_PASSWORD=\"$DEFAULT_ROOT_PASSWORD\""
fi

# Add the Podman image with the registry
PODMAN_RUN_CMD="$PODMAN_RUN_CMD docker.io/openmined/grid-backend:$VERSION"

# Debug output to verify Podman run command
echo "PODMAN_RUN_CMD: $PODMAN_RUN_CMD"

# Execute the Podman run command
eval $PODMAN_RUN_CMD
