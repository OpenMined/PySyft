#!/bin/bash

# Check if 1password CLI is installed
if ! command -v op &> /dev/null; then
    echo "1password CLI is not installed. Please install it before proceeding."
    echo "Installation instructions: https://support.1password.com/command-line-getting-started/"
    exit 1
fi

# Check if 1password CLI is logged in
if ! op vault list &> /dev/null; then
    echo "1password CLI is not logged in. Please log in using 'op signin' before proceeding."
    exit 1
fi

# Check if kube krew is installed
if ! command -v kubectl krew &> /dev/null; then
    echo "kube krew is not installed. Do you want to install it? (y/n)"
    read -r install_krew
    if [ "$install_krew" == "y" ]; then
        (
            set -x
            cd "$(mktemp -d)" &&
            OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
            ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/armv7l/arm/' -e 's/aarch64/arm64/')" &&
            curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/krew.tar.gz" &&
            tar zxvf krew.tar.gz &&
            KREW=./krew-"$OS"_"$ARCH" &&
            "$KREW" install krew
        )
        export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"
    else
        echo "kube krew is required for this script to work. Exiting."
        exit 1
    fi
fi

# Check if "kubectl passman" plugin is installed
if ! kubectl krew list | grep -q '^passman$'; then
    echo "kubectl passman plugin is not installed. Do you want to install it? (y/n)"
    read -r install_passman
    if [ "$install_passman" == "y" ]; then
        kubectl krew install passman
    else
        echo "kubectl passman plugin is required for this script to work. Exiting."
        exit 1
    fi
fi

# Prompt for the name
read -p "Enter the user name : " name

# Run the kubectl and jq command with the provided name
config_token=$(kubectl config view --raw -o json | jq --arg name "$name" '.users[] | select(.name==$name) | .user' -c)

# Check if the config_token is empty
if [ -z "$config_token" ]; then
  echo "Error: No configuration token found for name '$name'"
  exit 1
fi

# Set the CONFIG_TOKEN environment variable
export CONFIG_TOKEN="$config_token"


# Execute the specified command with CONFIG_TOKEN as an argument
kubectl passman 1password kubectl-prod-user "$CONFIG_TOKEN"
