#! /bin/bash

set -e

HELM_REPO="openmined/syft"
DATASITE_NAME="test-datasite"
KUBE_NAMESPACE="syft"
KUBE_CONTEXT=${KUBE_CONTEXT:-"k3d-syft-dev"}

UPGRADE_TYPE=$1

PROD="openmined/syft"
BETA="openmined/syft --devel"
DEV="./helm/syft"

if [ "$UPGRADE_TYPE" == "ProdToBeta" ]; then
    INSTALL_SOURCE=$PROD   # latest published prod
    UPGRADE_SOURCE=$BETA   # latest published beta
    INSTALL_ARGS=""
    UPGRADE_ARGS=""
elif [ "$UPGRADE_TYPE" == "BetaToDev" ]; then
    INSTALL_SOURCE=$BETA   # latest published beta
    UPGRADE_SOURCE=$DEV    # local chart
    INSTALL_ARGS=""
    UPGRADE_ARGS=""
elif [ "$UPGRADE_TYPE" == "ProdToDev" ]; then
    INSTALL_SOURCE=$PROD   # latest published prod
    UPGRADE_SOURCE=$DEV    # local chart
    INSTALL_ARGS=""
    UPGRADE_ARGS=""
else
    echo Invalid upgrade type $UPGRADE_TYPE
    exit 1
fi

kubectl config use-context $KUBE_CONTEXT
kubectl delete namespace syft || true
helm repo add openmined https://openmined.github.io/PySyft/helm
helm repo update openmined

echo Installing syft...
helm install $DATASITE_NAME $INSTALL_SOURCE $INSTALL_ARGS --namespace $KUBE_NAMESPACE --create-namespace
helm ls -A

WAIT_TIME=5 bash ./scripts/wait_for.sh service backend --namespace $KUBE_NAMESPACE
WAIT_TIME=5 bash ./scripts/wait_for.sh pod default-pool-0 --namespace $KUBE_NAMESPACE

echo Upgrading syft...
helm upgrade $DATASITE_NAME $UPGRADE_SOURCE $UPGRADE_ARGS --namespace $KUBE_NAMESPACE
helm ls -A

echo "Post-upgrade sleep" && sleep 5
WAIT_TIME=5 bash ./scripts/wait_for.sh service backend --namespace $KUBE_NAMESPACE
