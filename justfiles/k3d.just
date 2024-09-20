
# ---------------------------------------------------------------------------------------------------------------------

# Start a local registry on http://k3d-registry.localhost:{{port_registry}}
[group('registry')]
start-registry:
    k3d --version
    @-docker volume create k3d-registry-vol
    @-k3d registry create registry.localhost --port {{ port_registry }} -v k3d-registry-vol:/var/lib/registry --no-help

    if ! grep -q k3d-registry.localhost /etc/hosts; then \
        sudo {{ python_path }} scripts/patch_hosts.py --add-k3d-registry --fix-docker-hosts; \
    fi

    @curl --silent --retry 5 --retry-all-errors http://k3d-registry.localhost:{{ port_registry }}/v2/_catalog | jq
    @echo "\033[1;32mRegistring running at http://k3d-registry.localhost:{{ port_registry }}\033[0m"

[group('registry')]
delete-registry:
    -k3d registry delete registry.localhost
    -docker volume rm k3d-registry-vol

# ---------------------------------------------------------------------------------------------------------------------

# List all clusters
[group('cluster')]
list-clusters:
    k3d cluster list

# Stop all clusters
[group('cluster')]
delete-clusters:
    k3d cluster delete --all

[group('cluster')]
[private]
create-cluster cluster port *args='': start-registry && (apply-coredns cluster) (apply-signoz-agent cluster)
    #!/bin/bash
    set -euo pipefail

    k3d cluster create {{cluster}} \
        --port {{ port }}:80@loadbalancer \
        --registry-use k3d-registry.localhost:5800 {{ args }}

    # Since k3d adds k3d- prefix to the cluster name
    # we create a new context without the prefix
    kubectl config set-context {{ cluster }} --cluster=k3d-{{ cluster}} --user=admin@k3d-{{ cluster}}

[group('cluster')]
[private]
delete-cluster *args='':
    #!/bin/bash
    set -euo pipefail

    k3d cluster delete {{ args }}

[group('cluster')]
[private]
delete-ns context namespace:
    kubectl delete ns {{ namespace }} --force --grace-period=0 --context {{ context }}

[group('cluster')]
[private]
apply-coredns cluster:
    @echo "Applying custom CoreDNS config"

    kubectl apply -f ./scripts/k8s-coredns-custom.yml --context {{ cluster }}
    kubectl delete pod -n kube-system -l k8s-app=kube-dns --context {{ cluster }}

# ---------------------------------------------------------------------------------------------------------------------