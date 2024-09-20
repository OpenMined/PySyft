# ---------------------------------------------------------------------------------------------------------------------

# Reset Syft state in a cluster
# TODO: make reset_k8s.sh take in context and namespace as args
[group('utils')]
[private]
reset-syft name namespace:
    kubectl config use-context {{ name }}
    scripts/reset_k8s.sh

# K9s into the Datasite High cluster
[group('utils')]
k9s-high:
    k9s --context {{ cluster_high }}

# K9s into the Datesite Low cluster
[group('utils')]
k9s-low:
    k9s --context {{ cluster_low }}

# K9s into the Gateway cluster
[group('utils')]
k9s-gw:
    k9s --context {{ cluster_gw }}

# K9s into the Signoz cluster
[group('utils')]
k9s-signoz:
    k9s --context {{ cluster_signoz }}

# Stop all Syft clusters + registry
[group('utils')]
delete-all: delete-clusters delete-registry
    @echo "Stopped all Syft components"

[confirm('Confirm prune all docker resources?')]
[group('utils')]
prune-docker:
    -docker container prune -f
    -docker volume prune -af
    -docker image prune -af
    -docker builder prune -af
    -docker buildx prune -af
    -docker system prune -af --volumes

[group('utils')]
yank-ns namespace:
    -kubectl delete ns {{ namespace }} --now --timeout=5s
    kubectl get ns {{ namespace }} -o json | jq '.spec.finalizers = []' | kubectl replace --raw /api/v1/namespaces/{{ namespace }}/finalize -f -


# ---------------------------------------------------------------------------------------------------------------------