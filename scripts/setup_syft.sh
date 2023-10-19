#!/bin/bash

# Check if the required arguments are provided
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <version> <node_name> <node_side_type> <port>"
    exit 1
fi

VERSION=$1
NODE_NAME=$2
NODE_SIDE_TYPE=$3
PORT=$4
FILE_ID="1ffWtZa-aJkJYsAG8Wkosmvrt3Uvm1bYM" 

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
# Update the VERSION, NODE_NAME, NODE_SIDE_TYPE, and PORT values based on the OS
if [[ "$OS" == "mac" ]]; then
    sed -i '' "s/^VERSION=.*$/VERSION=$VERSION/" .env
    sed -i '' "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
    sed -i '' "s/^NODE_SIDE_TYPE=.*$/NODE_SIDE_TYPE=$NODE_SIDE_TYPE/" .env
    sed -i '' "s/^PORT=.*$/PORT=$PORT/" .env
else
    sed -i "s/^VERSION=.*$/VERSION=$VERSION/" .env
    sed -i "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
    sed -i "s/^NODE_SIDE_TYPE=.*$/NODE_SIDE_TYPE=$NODE_SIDE_TYPE/" .env
    sed -i "s/^PORT=.*$/PORT=$PORT/" .env
fi

# Run the docker compose command
docker-compose --env-file ./.env -p "$NODE_NAME" --profile blob-storage --profile frontend --file docker-compose.yml up -d

# Clean up the downloaded file
rm -f file.tgz

# Run the docker command for the Jupyter notebook using the specified PORT
#docker run --rm -it --network=host "openmined/grid-node-jupyter:$VERSION" "jupyter" "notebook" "--port=$PORT" "--ip=0.0.0.0" "--allow-root"
