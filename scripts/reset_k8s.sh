#!/bin/bash

KUBECTL_ARGS="$@"
NAMESPACE="syft"
POSTGRES_POD_NAME="postgres-0"

# if kubectl args doesn't have a namespace, add it
if [[ ! "$KUBECTL_ARGS" =~ (-n|--namespace) ]]; then
    KUBECTL_ARGS="$KUBECTL_ARGS --namespace $NAMESPACE"
fi

# SQL commands to reset all tables
RESET_COMMAND="
DO \$\$
DECLARE
    r RECORD;
BEGIN
    -- Disable all triggers
    SET session_replication_role = 'replica';

    -- Truncate all tables in the current schema
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = current_schema()) LOOP
        EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;

    -- Re-enable all triggers
    SET session_replication_role = 'origin';
END \$\$;

-- Reset all sequences
DO \$\$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = current_schema()) LOOP
        EXECUTE 'ALTER SEQUENCE ' || quote_ident(r.sequence_name) || ' RESTART WITH 1';
    END LOOP;
END \$\$;
"

# Execute the SQL commands
echo ">>> Resetting database '$POSTGRES_POD_NAME'. psql output:"
kubectl exec $KUBECTL_ARGS -i $POSTGRES_POD_NAME -- psql -U syft_postgres -d syftdb_postgres << EOF
$RESET_COMMAND
EOF

# Deleting StatefulSets that end with -pool
POOL_STATEFULSETS=$(kubectl get statefulsets $KUBECTL_ARGS -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep -E ".*-pool$")
if [ -n "$POOL_STATEFULSETS" ]; then
    echo ">>> Deleting '$POOL_STATEFULSETS'"
    for STATEFULSET in $POOL_STATEFULSETS; do
        kubectl delete statefulsets $KUBECTL_ARGS $STATEFULSET
        kubectl delete pods $KUBECTL_ARGS -l "app.kubernetes.io/component=$STATEFULSET" --grace-period=0 --force
    done
fi

# Resetting the backend pod
BACKEND_POD=$(kubectl get pods $KUBECTL_ARGS -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep -E ".*backend.*")
if [ -n "$BACKEND_POD" ]; then
    echo ">>> Re-creating '$BACKEND_POD'"
    kubectl delete pod $KUBECTL_ARGS $BACKEND_POD --grace-period=0 --force

    # wait for backend to come back up
    echo ">>> Waiting for '$BACKEND_POD' to be ready..."
    export WAIT_TIME=5
    bash packages/grid/scripts/wait_for.sh service backend $KUBECTL_ARGS > /dev/null
fi

echo ">>> Done"
