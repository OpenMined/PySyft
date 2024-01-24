#!/bin/bash

# # Check if the required arguments are provided
# if [[ $# -lt 5 ]]; then
#     echo "Usage: $0 <version> <node_name> <node_side_type> <node_type> <port>"
#     exit 1
# fi

# VERSION=$1
# NODE_NAME=$2
# NODE_SIDE_TYPE=$3
# NODE_TYPE=$4
# PORT=$5
# FILE_ID="1ffWtZa-aJkJYsAG8Wkosmvrt3Uvm1bYM"

# # Check if gdown is installed, if not, install it using pip
# if ! command -v gdown &> /dev/null; then
#     echo "gdown not found! Installing gdown..."
#     pip install gdown
# fi

# # Download the .tgz file using gdown
# gdown "$FILE_ID" -O file.tgz

# # Unzip the .tgz file
# tar -xzf file.tgz

# #Change directory to the unzipped folder
# cd syft-compose-files

# # Detect OS
# OS="linux"
# case "$(uname)" in
#     Darwin)
#         OS="mac"
#         ;;
# esac

# # Assuming the .env file is in the current directory after unzipping
# # Update the VERSION, NODE_NAME, NODE_SIDE_TYPE, NODE_TYPE and PORT values based on the OS
# if [[ "$OS" == "mac" ]]; then
#     sed -i '' "s/^VERSION=.*$/VERSION=$VERSION/" .env
#     sed -i '' "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
#     sed -i '' "s/^NODE_SIDE_TYPE=.*$/NODE_SIDE_TYPE=$NODE_SIDE_TYPE/" .env
#     sed -i '' "s/^NODE_TYPE=.*$/NODE_TYPE=$NODE_TYPE/" .env
#     sed -i '' "s/^PORT=.*$/PORT=$PORT/" .env
# else
#     sed -i "s/^VERSION=.*$/VERSION=$VERSION/" .env
#     sed -i "s/^NODE_NAME=.*$/NODE_NAME=$NODE_NAME/" .env
#     sed -i "s/^NODE_SIDE_TYPE=.*$/NODE_SIDE_TYPE=$NODE_SIDE_TYPE/" .env
#     sed -i "s/^NODE_TYPE=.*$/NODE_TYPE=$NODE_TYPE/" .env
#     sed -i "s/^PORT=.*$/PORT=$PORT/" .env
# fi

# # Run the docker compose command
# docker compose --env-file ./.env -p "$NODE_NAME" --profile blob-storage --profile frontend --file docker-compose.yml up -d

# # Clean up the downloaded file
# rm -f file.tgz

# Run the docker command for the Jupyter notebook using the specified PORT
##docker run --rm -it --network=host "openmined/grid-node-jupyter:$VERSION" "jupyter" "notebook" "--port=$PORT" "--ip=0.0.0.0" "--allow-root"

#Version 2

# Initialize default values
VERSION=""
NODE_NAME=""
NODE_SIDE_TYPE="high" # Default value for NODE_SIDE_TYPE
NODE_TYPE=""
PORT=""
USE_K3D=false
FILE_ID="1ffWtZa-aJkJYsAG8Wkosmvrt3Uvm1bYM"
TGZ_FILE="file.tgz"
COMPOSE_FILE="syft-compose-files"
CLUSTER_NAME="syft-test" # Default name for K3d cluster
NAMESPACE="syft" # Default namespace for K3d mode


# Function to display usage
# usage() {
#     echo "Usage: $0 [--use-k3d] [--version <version>] [--cluster-name <cluster_name>]"
#     echo "       $0 [-v|--version <version>] [-n|--name <node_name>] [-s|--side <node_side_type>] [-t|--type <node_type>] [-p|--port <port>]"
#     exit 1
# }

usage() {
    if [[ "$USE_K3D" = true ]]; then
        echo "Usage for K3d mode:"
        echo "$0 --use-k3d --version <version> [--cluster-name <cluster_name>] [--namespace <namespace>]"
    else
        echo "Usage for Docker mode:"
        echo "$0 [-v|--version <version>] [-n|--name <node_name>] [-s|--side <node_side_type>] [-t|--type <node_type>] [-p|--port <port>]"
    fi
    exit 1
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --use-k3d)
            USE_K3D=true
            shift
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -n|--name)
            [[ "$USE_K3D" = false ]] && NODE_NAME="$2"
            shift 2
            ;;
        -s|--side)
            [[ "$USE_K3D" = false ]] && NODE_SIDE_TYPE="$2"
            shift 2
            ;;
        -t|--type)
            [[ "$USE_K3D" = false ]] && NODE_TYPE="$2"
            shift 2
            ;;
        -p|--port)
            [[ "$USE_K3D" = false ]] && PORT="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Check if all required options are set (except NODE_SIDE_TYPE which has a default)
if [[ "$USE_K3D" = false && (-z "$VERSION" || -z "$NODE_NAME" || -z "$NODE_TYPE" || -z "$PORT") ]]; then
    echo "All Docker options except --side are required."
    usage
fi

if [[ "$USE_K3D" = true && -z "$VERSION" ]]; then
    echo "--version is required for K3d setup."
    usage
fi

# Conditional execution based on whether K3d is to be used
if [[ "$USE_K3D" = true ]]; then

     # Check if the cluster already exists
    if k3d cluster list | grep -qw "$CLUSTER_NAME"; then
        echo "Deleting existing K3d cluster named $CLUSTER_NAME"
        k3d cluster delete "$CLUSTER_NAME"
    fi


    # Step 1: Create a cluster with K3d
    echo "Creating K3d cluster named $CLUSTER_NAME"
    k3d cluster create "$CLUSTER_NAME" -p 8080:80@loadbalancer

    # Step 2: Use Helm to install Syft
    echo "Setting up Helm for Syft"
    helm repo add openmined https://openmined.github.io/PySyft/helm
    helm repo update

    # Step 3: Search for available Syft versions
    echo "Searching for available Syft versions"
    helm search repo openmined/syft --versions --devel

    # Step 4: Set your preferred Syft Chart version
    echo "Selected Syft version: $VERSION"

    # Step 5: Provisioning Helm Charts
    echo "Provisioning Helm Charts for Syft"
    helm install my-domain openmined/syft --version "$VERSION" --namespace "$NAMESPACE" --create-namespace

    else
        # Check and remove existing syft-compose file and file.tgz if they exist
            [ -f "$TGZ_FILE" ] && rm -f "$TGZ_FILE"
            [ -f "$COMPOSE_FILE" ] && rm -f "$COMPOSE_FILE"


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
            cd .. && rm -f file.tgz

    fi