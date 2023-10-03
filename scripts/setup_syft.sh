#!/bin/bash

# Check if the required arguments are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <version> <node_name>"
    exit 1
fi

VERSION=$1
NODE_NAME=$2
FILE_ID="1ffWtZa-aJkJYsAG8Wkosmvrt3Uvm1bYM" # Replace with your actual file ID

# Check if gdown is installed, if not, install it using pip
if ! command -v gdown &> /dev/null; then
    echo "gdown not found! Installing gdown..."
    pip install gdown
fi

# Download the .tgz file using gdown
gdown "$FILE_ID" -O file.tgz

# Unzip the .tgz file
tar -xzf file.tgz

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
# Update the VERSION and NODE_NAME values based on the OS
if [[ "$OS" == "mac" ]]; then
    sed -i '' "s/^VERSION=.*$/VERSION=$VERSION/" .env
    sed -i '' "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
else
    sed -i "s/^VERSION=.*$/VERSION=$VERSION/" .env
    sed -i "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
fi

# Run the docker compose command
docker compose --env-file ./.env -p "$NODE_NAME" --profile blob-storage --profile frontend --file docker-compose.yml up -d

# Clean up the downloaded file
rm -f file.tgz

