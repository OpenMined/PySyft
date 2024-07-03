#!/bin/bash

# Initialize default values
VERSION=""
NODE_NAME=""
NODE_SIDE_TYPE="high"
NODE_TYPE=""
PORT=""

# Function to display usage
usage() {
    echo "Usage: $0 [-v|--version <version>] [-n|--name <node_name>] [-s|--side <node_side_type>] [-t|--type <node_type>] [-p|--port <port>]"
    exit 1
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
        *)
            usage
            ;;
    esac
done

# Check if all required options are set
if [[ -z "$VERSION" || -z "$NODE_NAME" || -z "$NODE_TYPE" || -z "$PORT" ]]; then
    echo "All options are required."
    usage
fi


[ -f "$TGZ_FILE" ] && rm -f "$TGZ_FILE"
[ -f "$COMPOSE_FILE" ] && rm -f "$COMPOSE_FILE"


#Use curl to download the fille from azure blob storage
curl -L -o syft-file.tgz "https://openminedblob.blob.core.windows.net/syft-files/syft-compose-file.tar.gz"


# Unzip the .tgz file
tar -xzf syft-file.tgz

#Change directory to the unzipped folder
cd syft-compose-files


# Detect OS
OS="linux"
case "$(uname)" in
    Darwin)
        OS="mac"
        ;;
esac

# Assuming the .env file is in the current directory after unzipping
# Update the VERSION, NODE_NAME, NODE_SIDE_TYPE, NODE_TYPE, and PORT values based on the OS
if [[ "$OS" == "mac" ]]; then
    sed -i '' "s/^VERSION=.*$/VERSION=$VERSION/" .env
    sed -i '' "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
    sed -i '' "s/^NODE_SIDE_TYPE=.*$/NODE_SIDE_TYPE=$NODE_SIDE_TYPE/" .env
    sed -i '' "s/^NODE_TYPE=.*$/NODE_TYPE=$NODE_TYPE/" .env
    sed -i '' "s/^PORT=.*$/PORT=$PORT/" .env
else
    sed -i "s/^VERSION=.*$/VERSION=$VERSION/" .env
    sed -i "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
    sed -i "s/^NODE_SIDE_TYPE=.*$/NODE_SIDE_TYPE=$NODE_SIDE_TYPE/" .env
    sed -i "s/^NODE_TYPE=.*$/NODE_TYPE=$NODE_TYPE/" .env
    sed -i "s/^PORT=.*$/PORT=$PORT/" .env
fi



# Modify docker-compose.yml if the version is not 0.8.2-beta.6
if [[ "$VERSION" != "0.8.2-beta.6" ]]; then
    if [[ "$OS" == "mac" ]]; then
        sed -i '' '/command: "\/app\/grid\/start.sh"/s/^/#/' docker-compose.yml
    else
        sed -i '/command: "\/app\/grid\/start.sh"/s/^/#/' docker-compose.yml
    fi
fi



# Run the docker compose command
docker compose --env-file ./.env -p "$NODE_NAME" --profile blob-storage --profile frontend --file docker-compose.yml up -d

# Change directory out and clean up the downloaded file.tgz
cd .. && rm -f syft-file.tgz