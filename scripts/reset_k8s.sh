#!/bin/bash

echo $1

# Default pod name
DEFAULT_POD_NAME="postgres-0"

# Use the provided pod name or the default
POSTGRES_POD_NAME=${1:-$DEFAULT_POD_NAME}

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
echo "Resetting all tables in $POSTGRES_POD_NAME..."
kubectl exec -i -n syft $POSTGRES_POD_NAME -- psql -U syft_postgres -d syftdb_postgres << EOF
$RESET_COMMAND
EOF

echo "All tables in $POSTGRES_POD_NAME have been reset."

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
