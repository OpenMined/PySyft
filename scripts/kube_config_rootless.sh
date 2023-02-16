#!/bin/bash

# Add the following to ~/.bashrc after running this script
# export KUBECONFIG="${HOME}/.kube/config"

# ${HOME} is necessary. kubectl expects full path for KUBECONFIG
KUBECONFIG="${HOME}/.kube/config"
mkdir ~/.kube 2> /dev/null
sudo k3s kubectl config view --raw > $KUBECONFIG
chmod 600 $KUBECONFIG
