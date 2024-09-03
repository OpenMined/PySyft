#!/bin/bash

# WARNING: this will drop the 'app' database in your mongo-0 instance in the syft namespace
echo $1

# Dropping the database on mongo-0
if [ -z $1 ]; then
    MONGO_POD_NAME="mongo-0"
else
    MONGO_POD_NAME=$1
fi

DROPCMD="<<EOF
use app;
db.dropDatabase();
EOF"

FLUSH_COMMAND="mongosh -u root -p example $DROPCMD"
echo "$FLUSH_COMMAND" | kubectl exec -i -n syft $MONGO_POD_NAME -- bash 2>&1

# Resetting the backend pod
BACKEND_POD=$(kubectl get pods -n syft -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep -E ".*backend.*")
if [ -n "$BACKEND_POD" ]; then
    kubectl delete pod -n syft $BACKEND_POD
    echo "Backend pod $BACKEND_POD has been deleted and will be restarted."
else
    echo "No backend pod found."
fi

# Deleting StatefulSets that end with -pool
POOL_STATEFULSETS=$(kubectl get statefulsets -n syft -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep -E ".*-pool$")
if [ -n "$POOL_STATEFULSETS" ]; then
    for STATEFULSET in $POOL_STATEFULSETS; do
        kubectl delete statefulset -n syft $STATEFULSET
        echo "StatefulSet $STATEFULSET has been deleted."
    done
else
    echo "No StatefulSets ending with '-pool' found."
fi

# wait for backend to come back up
bash packages/grid/scripts/wait_for.sh service backend --namespace syft
