#!/bin/bash

# Initialize default values
VERSION=""
NODE_NAME=""
NODE_SIDE_TYPE="high" # Default value for NODE_SIDE_TYPE
NODE_TYPE=""
PORT=""
FILE_ID="1ffWtZa-aJkJYsAG8Wkosmvrt3Uvm1bYM" #G-drive file ID for the docker-compose files

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

# Check if all required options are set (except NODE_SIDE_TYPE which has a default)
if [[ -z "$VERSION" || -z "$NODE_NAME" || -z "$NODE_TYPE" || -z "$PORT" ]]; then
    echo "All options except --side are required."
    usage
fi

# Check if gdown is installed, if not, install it using pip
if ! command -v gdown &> /dev/null; then
    echo "gdown not found! Installing gdown..."
    pip install gdown
fi

# Download the .tgz file using gdown
gdown "$FILE_ID" -O file.tgz

# Unzip the .tgz file
tar -xzf file.tgz

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

# Run the docker compose command
docker-compose --env-file ./.env -p "$NODE_NAME" --profile blob-storage --profile frontend --file docker-compose.yml up -d



# Clean up the downloaded file
cd .. && rm -f file.tgz #&& rm -rf syft-compose-files

# Run the docker command for the Jupyter notebook using the specified PORT
#docker run --rm -it --network=host "openmined/grid-node-jupyter:$VERSION" "jupyter" "notebook" "--port=$PORT" "--ip=0.0.0.0" "--allow-root"
