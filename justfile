# Rules for new commands
# - Start with a verb
# - Keep it short (max. 3 words)
# - Group commands by context. Include group name in the command name.
# - Mark things private that are util functions with [private] or _var
# - Don't over-engineer, keep it simple.
# - Don't break existing commands

set dotenv-load

# ---------------------------------------------------------------------------------------------------------------------
# K3D cluster names
# Note: These are private (_ prefix) because we don't want it to be editable from CLI.
_name_default := "syft-dev"
_name_high := "syft-high"
_name_low := "syft-low"
_name_gw := "syft-gw"
_name_signoz := "signoz"

# K3D Registry name is used only in k3d.
_name_registry := "registry.localhost"

# Kubernetes namespaces for the deployments
# Note: These are private (_ prefix) because we don't want it to be editable from CLI.
_ns_default := "syft"
_ns_high := "high"
_ns_low := "low"
_ns_gw := "gw"

# Kubernetes context names generated for the K3D clusters
# Note: These are private (_ prefix) because we don't want it to be editable from CLI.
_ctx_default := "k3d-" + _name_default
_ctx_high := "k3d-" + _name_high
_ctx_low := "k3d-" + _name_low
_ctx_gw := "k3d-" + _name_gw
_ctx_signoz := "k3d-" + _name_signoz

# ---------------------------------------------------------------------------------------------------------------------

# Static Ports for the clusters
port_default := "8080"
port_high := port_default
port_low := "8081"
port_gw := "8082"
port_signoz_ui := "3301"
port_signoz_otel := "4317"
port_registry := "5800"

# Registry URL is used for
#   - setting up the registry for k3d clusters
#   - setting up the --var CONTAINER_REGISTRY for devspace deployments
# Note: Do not add http:// or https:// prefix
registry_url := "k3d-" + _name_registry + ":" + port_registry

# Signoz OTel endpoint is used for setting up the Otel collector
signoz_otel_url := "http://host.k3d.internal:" + port_signoz_otel

# ---------------------------------------------------------------------------------------------------------------------
# devspace profiles (comma-separated)
profiles := ""

# enable tracing by adding "tracing" profile in devspace
tracing := "true"

# add tracing profile if enabled
# This is private ( _prefix) to have a simple `just tracing=true ...`
_g_profiles := if tracing == "true" { profiles + ",tracing" } else { profiles }

# ---------------------------------------------------------------------------------------------------------------------
# this might break if you have alias python = python3 or either of the executable not pointing to the correct one
# just fix your system instead of making of fixing this
python_path := `which python || which python3`

# ---------------------------------------------------------------------------------------------------------------------

@default:
    just --list

# ---------------------------------------------------------------------------------------------------------------------

# Start a local registry on http://k3d-registry.local:5800 (port_registry=5800 or registry_url="gcr.io/path/to/registry")
[group('registry')]
start-registry:
    k3d --version
    @-docker volume create k3d-registry-vol
    @-k3d registry create {{ _name_registry }} --port {{ port_registry }} -v k3d-registry-vol:/var/lib/registry --no-help

    if ! grep -q {{ _name_registry }} /etc/hosts; then \
        sudo {{ python_path }} scripts/patch_hosts.py --add-k3d-registry --fix-docker-hosts; \
    fi

    @curl --silent --retry 5 --retry-all-errors http://{{ registry_url }}/v2/_catalog | jq
    @echo "\033[1;32mRegistring running at http://{{ registry_url }}\033[0m"

[group('registry')]
delete-registry:
    -k3d registry delete {{ _name_registry }}
    -docker volume rm k3d-registry-vol

# ---------------------------------------------------------------------------------------------------------------------

# Launch a Datasite high-side cluster on http://localhost:8080 (port_high=8080)
[group('highside')]
start-high: (create-cluster _name_high port_high)

# Stop the Datasite high-side cluster
[group('highside')]
delete-high: (delete-cluster _name_high)

# Deploy Syft to the high-side cluster
[group('highside')]
deploy-high: (deploy-devspace _ctx_high _ns_default)

# Reset Syft DB state in the high-side cluster
[group('highside')]
reset-high: (reset-syft _ctx_high _ns_default)

# Remove namespace from the high-side cluster
[group('highside')]
cleanup-high: (yank-ns _ctx_high _ns_default)

[group('highside')]
wait-high: (wait-pods _ctx_high _ns_default)

# K9s into the Datasite High cluster
[group('highside')]
k9s-high:
    k9s --context {{ _ctx_high }}

# ---------------------------------------------------------------------------------------------------------------------

# Launch a Datasite low-side cluster on http://localhost:8081 (port_low=8081)
[group('lowside')]
start-low: (create-cluster _name_low port_low)

# Stop the Datasite low-side cluster
[group('lowside')]
delete-low: (delete-cluster _name_low)

# Deploy Syft to the low-side cluster
[group('lowside')]
deploy-low: (deploy-devspace _ctx_low _ns_default "-p datasite-low")

# Reset Syft DB state in the low-side cluster
[group('lowside')]
reset-low: (reset-syft _ctx_low _ns_default)

# Remove namespace from the low-side cluster
[group('lowside')]
cleanup-low: (yank-ns _ctx_low _ns_default)

[group('lowside')]
wait-low: (wait-pods _ctx_low _ns_default)

# K9s into the Datesite Low cluster
[group('lowside')]
k9s-low:
    k9s --context {{ _ctx_low }}

# ---------------------------------------------------------------------------------------------------------------------

# Launch a Gateway cluster on http://localhost:8083 (port_gw=8083)
[group('gateway')]
start-gw: (create-cluster _name_gw port_gw)

# Delete the Gateway cluster
[group('gateway')]
delete-gw: (delete-cluster _name_gw)

# Deploy Syft to the gateway cluster
[group('gateway')]
deploy-gw: (deploy-devspace _ctx_gw _ns_default "-p gateway")

# Reset Syft DB state in the gateway cluster
[group('gateway')]
reset-gw: (reset-syft _ctx_gw _ns_default)

# Remove namespace from the gateway cluster
[group('gateway')]
cleanup-gw: (yank-ns _ctx_gw _ns_default)

[group('gateway')]
wait-gw: (wait-pods _ctx_gw _ns_default)

# K9s into the Gateway cluster
[group('gateway')]
k9s-gw:
    k9s --context {{ _ctx_gw }}

# ---------------------------------------------------------------------------------------------------------------------

# Launch SigNoz. UI=http://localhost:3301 OTEL=http://localhost:4317 (port_signoz_ui=3301 port_signoz_otel=4317)
[group('signoz')]
start-signoz: && (apply-signoz _ctx_signoz) (setup-signoz _ctx_signoz)
    k3d cluster create {{ _name_signoz }} \
        --port {{ port_signoz_ui }}:3301@loadbalancer \
        --port {{ port_signoz_otel }}:4317@loadbalancer \
        --k3s-arg "--disable=metrics-server@server:*"

# Remove SigNoz from the cluster
[group('signoz')]
delete-signoz: (delete-cluster _name_signoz)

# Remove all SigNoz data without deleting
[group('signoz')]
reset-signoz:
    @kubectl exec --context {{ _ctx_signoz }} -n platform chi-signoz-clickhouse-cluster-0-0-0 --container clickhouse -- \
        clickhouse-client --multiline --multiquery "\
        TRUNCATE TABLE signoz_analytics.rule_state_history_v0; \
        TRUNCATE TABLE signoz_logs.logs_v2; \
        TRUNCATE TABLE signoz_logs.logs; \
        TRUNCATE TABLE signoz_logs.usage; \
        TRUNCATE TABLE signoz_metrics.usage; \
        TRUNCATE TABLE signoz_traces.durationSort; \
        TRUNCATE TABLE signoz_traces.signoz_error_index_v2; \
        TRUNCATE TABLE signoz_traces.signoz_index_v2; \
        TRUNCATE TABLE signoz_traces.signoz_spans; \
        TRUNCATE TABLE signoz_traces.top_level_operations; \
        TRUNCATE TABLE signoz_traces.usage_explorer; \
        TRUNCATE TABLE signoz_traces.usage;"

    @echo "Done. Traces & logs are cleared, but graphs may still show old content."

# K9s into the Signoz cluster
[group('signoz')]
k9s-signoz:
    k9s --context {{ _ctx_signoz }}

[group('signoz')]
[private]
apply-collector kube_context:
    @echo "Installing SigNoz OTel Collector in kubernetes context {{ kube_context }}"
    helm install k8s-infra k8s-infra \
        --repo https://charts.signoz.io \
        --kube-context {{ kube_context }} \
        --set global.deploymentEnvironment=local \
        --set clusterName={{ kube_context }} \
        --set otelCollectorEndpoint={{ signoz_otel_url }} \
        --set otelInsecure=true \
        --set presets.otlpExporter.enabled=true \
        --set presets.loggingExporter.enabled=true

# Remove SigNoz from the cluster
[group('signoz')]
delete-collector:
    helm uninstall k8s-infra

[group('signoz')]
[private]
apply-signoz kube_context:
    @echo "Installing SigNoz in kube context {{ kube_context }}"
    helm install signoz signoz \
        --repo https://charts.signoz.io \
        --kube-context {{ kube_context }} \
        --namespace platform \
        --create-namespace \
        --version 0.52.0 \
        --set frontend.service.type=LoadBalancer \
        --set otelCollector.service.type=LoadBalancer \
        --set otelCollectorMetrics.service.type=LoadBalancer

[group('signoz')]
[private]
setup-signoz kube_context:
    #!/bin/bash
    set -euo pipefail

    SIGNOZ_URL="http://localhost:3301"
    USERNAME="admin@localhost"
    PASSWORD="password"
    DASHBOARDS=(
        "https://raw.githubusercontent.com/SigNoz/dashboards/refs/heads/main/k8s-infra-metrics/kubernetes-pod-metrics-detailed.json"
        "https://raw.githubusercontent.com/SigNoz/dashboards/refs/heads/main/k8s-infra-metrics/kubernetes-node-metrics-detailed.json"
        "https://raw.githubusercontent.com/SigNoz/dashboards/refs/heads/main/k8s-infra-metrics/kubernetes-cluster-metrics.json"
    )

    echo "Waiting for SigNoz frontend to be available..."
    bash ./packages/grid/scripts/wait_for.sh service signoz-frontend \
        --namespace platform --context {{ kube_context }} &> /dev/null

    echo "Setting up SigNoz account..."
    curl -s --retry 5 --retry-all-errors -X POST \
        -H "Content-Type: application/json" \
        --data "{\"email\":\"$USERNAME\",\"name\":\"admin\",\"orgName\":\"openmined\",\"password\":\"$PASSWORD\"}" \
        "$SIGNOZ_URL/api/v1/register"

    echo "Adding some dashboards..."
    AUTH_TOKEN=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" \
        "$SIGNOZ_URL/api/v1/login" | jq -r .accessJwt)

    if [ -z "$AUTH_TOKEN" ] || [ "$AUTH_TOKEN" = "null" ]; then
        echo "Could not set up dashboards. But you can do it manually from the dashboard."
        exit 0
    fi

    for URL in "${DASHBOARDS[@]}"; do
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH_TOKEN" \
            -d "$(curl -s --retry 3 --retry-all-errors "$URL")" \
            "$SIGNOZ_URL/api/v1/dashboards" &> /dev/null
    done

    printf "\nSignoz is ready and running on %s\n" "$SIGNOZ_URL"
    printf "Email: \033[1;36m%s\033[0m\n" "$USERNAME"
    printf "Password: \033[1;36m%s\033[0m\n" "$PASSWORD"

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
create-cluster cluster_name port *args='': start-registry && (apply-coredns "k3d-" + cluster_name) (apply-collector "k3d-" + cluster_name)
    k3d cluster create {{ cluster_name }} \
        --port {{ port }}:80@loadbalancer \
        --k3s-arg "--disable=metrics-server@server:*" \
        --registry-use {{ registry_url }} {{ args }}

[group('cluster')]
[private]
delete-cluster *args='':
    #!/bin/bash
    set -euo pipefail

    # remove the k3d- prefix
    ARGS=$(echo "{{ args }}" | sed -e 's/k3d-//g')
    k3d cluster delete $ARGS

[group('cluster')]
[private]
apply-coredns kube_context:
    @echo "Applying custom CoreDNS config"

    kubectl apply -f ./scripts/k8s-coredns-custom.yml --context {{ kube_context }}
    kubectl delete pod -n kube-system -l k8s-app=kube-dns --context {{ kube_context }}

# ---------------------------------------------------------------------------------------------------------------------

[group('devspace')]
[private]
deploy-devspace kube_context namespace *args='':
    #!/bin/bash
    set -euo pipefail

    cd packages/grid

    PROFILE="{{ _g_profiles }}"
    PROFILE=$(echo "$PROFILE" | sed -E 's/^,*|,*$//g')
    if [ -n "$PROFILE" ]; then
        PROFILE="-p $PROFILE"
    fi

    echo "Deploying to kube context {{ kube_context }}"

    devspace deploy -b \
        --no-warn \
        --kube-context {{ kube_context }} \
        --namespace {{ namespace }} \
        $PROFILE \
        {{ args }} \
        --var CONTAINER_REGISTRY={{ registry_url }}

[group('devspace')]
[private]
purge-devspace kube_context namespace:
    #!/bin/bash
    set -euo pipefail

    cd packages/grid
    devspace purge --force-purge --kube-context {{ kube_context }} --no-warn --namespace {{ namespace }}
    sleep 3

# ---------------------------------------------------------------------------------------------------------------------

[group('cloud')]
[private]
check-platform:
    #!/bin/bash
    set -euo pipefail

    OSTYPE=$(uname -sm)
    MSG="==================================================================================================\n\
    Deploying dev->cloud k8s (x64 nodes) requires images to be built with --platform=linux/amd64\n\
    On Apple Silicon, cross-platform image is unstable on different providers\n\n\
    Current status:\n\
    ✅ | Docker Desktop | 4.34.0+ | *Enable* containerd and *uncheck* 'Use Rosetta for x86_64/amd64...'\n\
    ❌ | OrbStack       | 1.7.2   | Rosetta: gets stuck & qemu: errors with 'illegal instruction'\n\
    ❌ | Lima VM/Colima | 0.23.2  | Rosetta: gets stuck & qemu: errors with 'illegal instruction'\n\
    =================================================================================================="

    if [[ "$OSTYPE" == "Darwin arm64" ]]; then
        echo -e $MSG
    fi

[group('cloud')]
[private]
deploy-cloud kube_context registry_url namespace profile: check-platform
    #!/bin/bash

    CONTEXT_NAME=$(kubectl config get-contexts -o=name | grep "{{ kube_context }}")

    if [ -z "$CONTEXT_NAME" ]; then
        echo "Context not found: {{ kube_context }}. Authorized with cloud providers to get relevant K8s cluster contexts"
        exit 1
    fi

    set -euo pipefail

    # cloud deployments always have tracing false + platform=amd64
    just tracing=false registry_url={{ registry_url }} \
        deploy-devspace $CONTEXT_NAME {{ namespace }} "-p {{ profile }} --var PLATFORM=amd64"

[group('cloud')]
[private]
purge-cloud kube_context namespace:
    #!/bin/bash

    CONTEXT_NAME=$(kubectl config get-contexts -o=name | grep "{{ kube_context }}")

    if [ -z "$CONTEXT_NAME" ]; then
        echo "Context not found: {{ kube_context }}. Authorized with cloud providers to get relevant K8s cluster contexts"
        exit 1
    fi

    set -euo pipefail

    just purge-devspace $CONTEXT_NAME {{ namespace }}
    kubectl delete ns {{ namespace }} --force --grace-period=0 --context $CONTEXT_NAME

# ---------------------------------------------------------------------------------------------------------------------

# Auth all components required for deploying Syft to Google Cloud
[group('cloud-gcp')]
auth-gcloud:
    #!/bin/bash
    set -euo pipefail

    # login to gcloud
    ACCOUNT=$(gcloud config get-value account)
    if [ -z "$ACCOUNT" ]; then
        gcloud auth login
    fi

    echo "Logged in as \"$(gcloud config get-value account)\""

    # install gke-gcloud-auth-plugin
    gke_installed=$(gcloud components list --only-local-state --filter gke-gcloud-auth-plugin --format=list 2>/dev/null)
    if [ -z "$gke_installed" ]; then
        gcloud components install gke-gcloud-auth-plugin
        echo "Installed gke-gcloud-auth-plugin"
    fi

# Deploy local code as datasite-high to Google Kubernetes Engine
[group('cloud-gcp')]
deploy-gcp-high gcp_cluster gcp_registry_url namespace="syft": (deploy-cloud gcp_cluster gcp_registry_url namespace "gcp")

# Deploy local code as datasite-high to Google Kubernetes Engine
[group('cloud-gcp')]
deploy-gcp-low gcp_cluster gcp_registry_url namespace="syft": (deploy-cloud gcp_cluster gcp_registry_url namespace "gcp-low")

# Purge deployment from a cluster
[group('cloud-gcp')]
purge-gcp gcp_cluster namespace="syft": (purge-cloud gcp_cluster namespace)

# ---------------------------------------------------------------------------------------------------------------------

[group('cloud-az')]
auth-az tenant="creditsopenmined.onmicrosoft.com":
    #!/bin/bash

    # login to azure
    ACCOUNT=$(az account show --query user.name)
    if [ -z "$ACCOUNT" ]; then
        az login --tenant {{ tenant }}
    fi

    echo "Logged in as $(az account show --query user.name)"

# Deploy local code as datasite-high to Azure Kubernetes Service
[group('cloud-az')]
deploy-az-high aks_cluster az_registry namespace="syft": (deploy-cloud aks_cluster az_registry namespace "azure")

# ---------------------------------------------------------------------------------------------------------------------

# Reset Syft state in a cluster
[group('utils')]
reset-syft kube_context namespace:
    scripts/reset_k8s.sh --context {{ kube_context }} --namespace {{ namespace }}

# Delete all local clusters and registry
[group('utils')]
delete-all: delete-clusters delete-registry

# Prune local docker cache. Run atleast once a month.
[group('utils')]
prune-docker:
    -docker container prune -f
    -docker volume prune -af
    -docker image prune -af
    -docker system prune -af --volumes

# Delete all resources in a namespace
[group('utils')]
yank-ns kube_context namespace:
    # delete pods 𝙛 𝙖 𝙨 𝙩
    -kubectl delete statefulsets --all --context {{ kube_context }} --namespace {{ namespace }} --now
    -kubectl delete deployments --all --context {{ kube_context }} --namespace {{ namespace }} --now
    -kubectl delete pods --all --namespace {{ namespace }} --grace-period=0 --force

    # delete resources 𝙛 𝙖 𝙨 𝙩
    -kubectl delete configmap --all --context {{ kube_context }} --namespace {{ namespace }} --now
    -kubectl delete secrets --all --context {{ kube_context }} --namespace {{ namespace }} --now
    -kubectl delete ingress --all --context {{ kube_context }} --namespace {{ namespace }} --now

    # delete namespace NOT 𝙛 𝙖 𝙨 𝙩 :(
    -kubectl delete ns {{ namespace }} --context {{ kube_context }} --grace-period=0 --force --timeout=5s

    # Too slow... yanking it
    -kubectl get ns {{ namespace }} --context {{ kube_context }} -o json | jq '.spec.finalizers = []' | \
        kubectl replace --context {{ kube_context }} --raw /api/v1/namespaces/{{ namespace }}/finalize -f -

    @echo "Done"

# Wait for all pods to be ready in a namespace
[group('utils')]
@wait-pods kube_context namespace:
    echo "Waiting for all pods to be ready in cluster={{ kube_context }} namespace={{ namespace }}"
    # Wait for at least one pod to appear (timeout after 5 minutes)
    timeout 300 bash -c 'until kubectl get pods --context {{ kube_context }} --namespace {{ namespace }} 2>/dev/null | grep -q ""; do sleep 5; done'

    kubectl wait --for=condition=ready pod --all --timeout=300s --context {{ kube_context }} --namespace {{ namespace }}

    # if the above doesn't wait as we expect the drop the above and use the below
    # @bash packages/grid/scripts/wait_for.sh service proxy --context {{ kube_context }} --namespace {{ namespace }}
    # @bash packages/grid/scripts/wait_for.sh service frontend --context {{ kube_context }} --namespace {{ namespace }}
    # @bash packages/grid/scripts/wait_for.sh service postgres --context {{ kube_context }} --namespace {{ namespace }}
    # @bash packages/grid/scripts/wait_for.sh service seaweedfs --context {{ kube_context }} --namespace {{ namespace }}
    # @bash packages/grid/scripts/wait_for.sh service backend --context {{ kube_context }} --namespace {{ namespace }}
    echo "All pods are ready"
