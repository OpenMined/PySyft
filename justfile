cluster_default := "k3d-syft-dev"
cluster_high := "k3d-syft-high"
cluster_low := "k3d-syft-low"
cluster_gw := "k3d-syft-gw"
cluster_signoz := "k3d-signoz"

port_default := "8080"
port_high := port_default
port_low := "8081"
port_gw := "8082"
port_signoz_ui := "3301"
port_signoz_otel := "4317"
port_registry := "5800"

ns_default := "syft"
ns_high := "high"
ns_low := "low"
ns_gw := "gw"

registry_url := "k3d-registry.localhost:" + port_registry
signoz_otel_url := "http://host.k3d.internal:" + port_signoz_otel

profiles := ""
tracing := "true"

_g_profiles := if tracing == "true" { profiles + ",tracing" } else { profiles }

python_path := `which python || which python3`

# ---------------------------------------------------------------------------------------------------------------------

@default:
    just --list

# ---------------------------------------------------------------------------------------------------------------------

# Start a local registry on http://k3d-registry.localhost:{{port_registry}}
[group('registry')]
start-registry:
    k3d --version
    @-docker volume create k3d-registry-vol
    @-k3d registry create registry.localhost --port {{ port_registry }} -v k3d-registry-vol:/var/lib/registry --no-help

    if ! grep -q k3d-registry.localhost /etc/hosts; then \
        sudo {{python_path}} scripts/patch_hosts.py --add-k3d-registry --fix-docker-hosts; \
    fi

    @curl --silent --retry 5 --retry-all-errors http://k3d-registry.localhost:{{ port_registry }}/v2/_catalog | jq
    @echo "\033[1;32mRegistring running at http://k3d-registry.localhost:{{ port_registry }}\033[0m"

[group('registry')]
stop-registry:
    -k3d registry delete registry.localhost
    -docker volume rm k3d-registry-vol

# ---------------------------------------------------------------------------------------------------------------------

# Launch Syft Datasite high-side cluster on http://localhost:{{port_high}}
[group('highside')]
start-high: (cluster-delete cluster_high) (cluster-create cluster_high port_high)
    @echo "Started Syft Datasite (high-side) on http://localhost:{{ port_high }}/"

# Stop Syft Datasite high-side cluster
[group('highside')]
stop-high: (cluster-delete cluster_high)
    @echo "Stopped Syft Datasite (high-side)"

[group('highside')]
deploy-high: (devspace-deploy cluster_high ns_default)
    @echo "Done"

[group('highside')]
reset-high: (state-reset cluster_high ns_default)
    @echo "Done"

[group('highside')]
cleanup-high: (devspace-purge cluster_high ns_default) && (ns-cleanup cluster_high ns_default)
    @echo "Done"

# ---------------------------------------------------------------------------------------------------------------------

# Launch Syft Datasite low-side cluster on http://localhost:{{port_low}}
[group('lowside')]
start-low: (cluster-create cluster_low port_low)
    @echo "Started Syft Datasite (low-side) on http://localhost:{{ port_low }}/"

# Stop Syft Datasite low-side cluster
[group('lowside')]
stop-low: (cluster-delete cluster_low)
    @echo "Stopped Syft Datasite (low-side)"

[group('lowside')]
deploy-low: (devspace-deploy cluster_low ns_default "-p datasite-low")
    @echo "Done"

[group('lowside')]
reset-low: (state-reset cluster_low ns_default)
    @echo "Done"

[group('lowside')]
cleanup-low: (devspace-purge cluster_low ns_default) && (ns-cleanup cluster_low ns_default)
    @echo "Done"

# ---------------------------------------------------------------------------------------------------------------------

# Launch Syft Gateway cluster on http://localhost:{{port_gw}}
[group('gateway')]
start-gw: (cluster-create cluster_gw port_gw)
    @echo "Started Syft Gateway on http://localhost:{{ port_gw }}/"

# Stop Syft Gateway cluster
[group('gateway')]
stop-gw: (cluster-delete cluster_gw)
    @echo "Stopped Syft Gateway"

[group('gateway')]
deploy-gw: (devspace-deploy cluster_gw ns_default "-p gateway")
    @echo "Done"

[group('gateway')]
reset-gw: (state-reset cluster_gw ns_default)
    @echo "Done"

[group('gateway')]
cleanup-gw: (devspace-purge cluster_gw ns_default) && (ns-cleanup cluster_gw ns_default)
    @echo "Done"

# ---------------------------------------------------------------------------------------------------------------------

# Launch a Syft shared cluster on http://localhost:{{port_default}}
[group('shared')]
start-shared: (cluster-create cluster_default port_default "--agents 2 --servers 1")
    @echo "Started Syft on http://localhost:{{ port_default }}/"

# Stop Syft shared cluster
[group('shared')]
stop-shared: (cluster-delete cluster_default)
    @echo "Stopped Syft cluster"

# Deploy to "high" namespace on the shared cluster
[group('shared')]
deploy-ns-high: (devspace-deploy cluster_default ns_high)
    @echo "Deployed Syft Gateway on {{ cluster_default }}"

# Delete the "high" namespace
[group('shared')]
delete-ns-high: (ns-cleanup cluster_default ns_high)
    @echo "Done"

# Deploy to "low" namespace on the shared cluster
[group('shared')]
deploy-ns-low: (devspace-deploy cluster_default ns_low "-p datasite-low")
    @echo "Deployed Syft Gateway on {{ cluster_default }}"

# Delete the "low" namespace
[group('shared')]
delete-ns-low: (ns-cleanup cluster_default ns_low)
    @echo "Done"

# Deploy to "gw" namespace on the shared cluster
[group('shared')]
deploy-ns-gw: (devspace-deploy cluster_default ns_gw "-p gateway")
    @echo "Deployed Syft Gateway on {{ cluster_default }}"

# Delete the "gw" namespace
[group('shared')]
delete-ns-gw: (ns-cleanup cluster_default ns_gw)
    @echo "Done"

# ---------------------------------------------------------------------------------------------------------------------

# Launch SigNoz on http://localhost:{{port_signoz_ui}}
[group('signoz')]
start-signoz: && apply-signoz setup-signoz
    k3d cluster create signoz \
        --port {{ port_signoz_ui }}:3301@loadbalancer \
        --port {{ port_signoz_otel }}:4317@loadbalancer \
        --k3s-arg "--disable=metrics-server@server:*"

    @printf "Started SigNoz\nDashboard: http://localhost:{{ port_signoz_ui }}\nOTEL Endpoint: http://localhost:{{ port_signoz_otel }}\n"

[group('signoz')]
delete-collector:
    helm uninstall k8s-infra

# Stop SigNoz cluster
[group('signoz')]
stop-signoz: (cluster-delete cluster_signoz)
    @echo "Stopped SigNoz cluster"

[group('signoz')]
[private]
apply-collector cluster:
    @echo "Installing SigNoz OTel Collector"
    helm install k8s-infra k8s-infra \
        --repo https://charts.signoz.io \
        --kube-context {{ cluster }} \
        --set global.deploymentEnvironment=local \
        --set clusterName={{ cluster }} \
        --set otelCollectorEndpoint={{ signoz_otel_url }} \
        --set otelInsecure=true \
        --set presets.otlpExporter.enabled=true \
        --set presets.loggingExporter.enabled=true

[group('signoz')]
[private]
apply-signoz:
    @echo "Installing SigNoz on the cluster"
    helm install signoz signoz \
        --repo https://charts.signoz.io \
        --kube-context {{ cluster_signoz }} \
        --namespace platform \
        --create-namespace \
        --version 0.52.0 \
        --set frontend.service.type=LoadBalancer \
        --set otelCollector.service.type=LoadBalancer \
        --set otelCollectorMetrics.service.type=LoadBalancer

[group('signoz')]
[private]
setup-signoz:
    @echo "Waiting for SigNoz frontend to be available..."
    @bash ./packages/grid/scripts/wait_for.sh service signoz-frontend --namespace platform --context {{ cluster_signoz }} &> /dev/null

    @echo "Setting up SigNoz account"
    @curl --retry 5 --retry-all-errors -X POST \
        -H "Content-Type: application/json" \
        --data '{"email":"admin@localhost","name":"admin","orgName":"openmined","password":"password"}' \
        http://localhost:3301/api/v1/register

    @printf '\nSignoz is running on http://localhost:3301\nEmail: \033[1;36madmin@localhost\033[0m\nPassword: \033[1;36mpassword\033[0m\n'

# ---------------------------------------------------------------------------------------------------------------------

[group('cluster')]
[private]
cluster-create cluster port *args='': start-registry && (apply-coredns cluster) (apply-collector cluster)
    #!/bin/bash
    set -euo pipefail

    # remove the k3d- prefix
    CLUSTER_NAME=$(echo "{{ cluster }}" | sed -e 's/k3d-//g')

    k3d cluster create $CLUSTER_NAME \
        --port {{ port }}:80@loadbalancer \
        --registry-use k3d-registry.localhost:5800 {{ args }}

[group('cluster')]
[private]
cluster-delete *args='':
    #!/bin/bash
    set -euo pipefail

    # remove the k3d- prefix
    ARGS=$(echo "{{ args }}" | sed -e 's/k3d-//g')
    k3d cluster delete $ARGS

[group('cluster')]
[private]
ns-cleanup context namespace:
    kubectl delete ns {{ namespace }} --force --grace-period=0 --context {{ context }}

[group('cluster')]
cluster-list:
    k3d cluster list

# Stop all Syft clusters
[group('cluster')]
stop-all: (cluster-delete cluster_default cluster_high cluster_low cluster_gw cluster_signoz)
    @echo "Stopped all Syft clusters"

[group('cluster')]
[private]
apply-coredns cluster:
    @echo "Applying custom CoreDNS config"

    kubectl apply -f ./scripts/k8s-coredns-custom.yml --context {{ cluster }}
    kubectl delete pod -n kube-system -l k8s-app=kube-dns --context {{ cluster }}

[group('cluster')]
k9s-high:
    k9s --context {{ cluster_high }}

[group('cluster')]
k9s-low:
    k9s --context {{ cluster_low }}

[group('cluster')]
k9s-gw:
    k9s --context {{ cluster_gw }}

[group('cluster')]
k9s-signoz:
    k9s --context {{ cluster_signoz }}

[group('cluster')]
k9s-shared:
    k9s --context {{ cluster_default }}

# ---------------------------------------------------------------------------------------------------------------------

[private]
devspace-deploy cluster namespace *args='':
    #!/bin/bash
    set -euo pipefail

    cd packages/grid

    PROFILE="{{ _g_profiles }}"
    PROFILE=$(echo "$PROFILE" | sed -E 's/^,*|,*$//g')
    if [ -n "$PROFILE" ]; then
        PROFILE="-p $PROFILE"
    fi

    devspace deploy -b \
        --no-warn \
        --kube-context {{ cluster }} \
        --namespace {{ namespace }} \
        $PROFILE \
        {{ args }} \
        --var CONTAINER_REGISTRY={{ registry_url }}

[private]
devspace-purge cluster namespace:
    #!/bin/bash
    set -euo pipefail

    cd packages/grid
    devspace purge --force-purge --kube-context {{ cluster }} --no-warn --namespace {{ namespace }}
    sleep 3

# ---------------------------------------------------------------------------------------------------------------------

[group('cloud')]
[private]
deploy-cloud cluster registry namespace profile:
    #!/bin/bash
    set -euo pipefail

    kubectl config get-contexts {{ cluster }} > /dev/null
    # cloud deployments always have tracing false + platform=amd64
    just tracing=false registry_url={{ registry }} devspace-deploy {{ cluster }} {{ namespace }} "-p {{ profile }} --var PLATFORM=amd64"

[group('cloud')]
deploy-gcp-high gke_cluster gcp_registry namespace="syft": (deploy-cloud gke_cluster gcp_registry namespace "gcp")

[group('cloud')]
deploy-gcp-low gke_cluster gcp_registry namespace="syft": (deploy-cloud gke_cluster gcp_registry namespace "gcp-low")

[group('cloud')]
deploy-az-high aks_cluster az_registry namespace="syft": (deploy-cloud aks_cluster az_registry namespace "azure")

# ---------------------------------------------------------------------------------------------------------------------

[group('syft')]
[private]
state-reset name namespace:
    kubectl config use-context {{ name }}
    scripts/reset_k8s.sh
