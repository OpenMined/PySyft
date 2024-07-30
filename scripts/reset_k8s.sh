#!/bin/bash

set -e

# Default context to current if not provided
CONTEXT=${1:-$(kubectl config current-context)}
NAMESPACE=${2:-syft}

echo "Resetting Kubernetes resources in context $CONTEXT and namespace $NAMESPACE"

print_progress() {
  echo -e "\033[1;32m$1\033[0m"
}

# Set the Kubernetes context
kubectl config use-context $CONTEXT

# Function to reset a StatefulSet and delete its PVCs
reset_statefulset() {
  local statefulset=$1
  local component=$2

  print_progress "Scaling down $statefulset StatefulSet..."
  kubectl scale statefulset $statefulset --replicas=0 -n $NAMESPACE

  print_progress "Deleting PVCs for $statefulset..."
  local pvcs=$(kubectl get pvc -l app.kubernetes.io/component=$component -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
  for pvc in $pvcs; do
    kubectl delete pvc $pvc -n $NAMESPACE
  done

  print_progress "Scaling up $statefulset StatefulSet..."
  kubectl scale statefulset $statefulset --replicas=1 -n $NAMESPACE

  print_progress "Waiting for $statefulset StatefulSet to be ready..."
  kubectl rollout status statefulset $statefulset -n $NAMESPACE
}

# Function to delete a StatefulSet
delete_statefulset() {
  local statefulset=$1

  print_progress "Deleting $statefulset StatefulSet..."
  kubectl delete statefulset $statefulset -n $NAMESPACE

#   # Since Default Pool does not have any PVCs, we can skip this step
#   print_progress "Deleting PVCs for $statefulset..."
#   local pvcs=$(kubectl get pvc -l statefulset.kubernetes.io/pod-name=${statefulset}-0 -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
#   for pvc in $pvcs; do
#     kubectl delete pvc $pvc -n $NAMESPACE
#   done

  print_progress "Waiting for $statefulset StatefulSet to be fully deleted..."
  kubectl wait --for=delete statefulset/$statefulset -n $NAMESPACE
}

# Reset MongoDB StatefulSet
reset_statefulset "mongo" "mongo"

# Reset SeaweedFS StatefulSet
reset_statefulset "seaweedfs" "seaweedfs"

# Delete default-pool StatefulSet
delete_statefulset "default-pool"

# Restart Backend StatefulSet
print_progress "Restarting backend StatefulSet..."
kubectl scale statefulset backend --replicas=0 -n $NAMESPACE
kubectl scale statefulset backend --replicas=1 -n $NAMESPACE
print_progress "Waiting for backend StatefulSet to be ready..."
kubectl rollout status statefulset backend -n $NAMESPACE


print_progress "All operations completed successfully."
