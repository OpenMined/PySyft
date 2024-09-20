# ---------------------------------------------------------------------------------------------------------------------

# Launch SigNoz on http://localhost:{{port_signoz_ui}}
[group('signoz')]
start-signoz: && apply-signoz setup-signoz
    k3d cluster create {{ cluster_signoz }} \
        --port {{ port_signoz_ui }}:3301@loadbalancer \
        --port {{ port_signoz_otel }}:4317@loadbalancer \
        --k3s-arg "--disable=metrics-server@server:*"

    # Since k3d adds k3d- prefix to the cluster name
    # we create a new context without the prefix
    kubectl config set-context {{ cluster_signoz }} --cluster=k3d-{{ cluster_signoz }} \
        --user=admin@k3d-{{ cluster_signoz }}

    @printf "Started SigNoz\n\
        Dashboard: \033[1;36mhttp://localhost:{{ port_signoz_ui }}\033[0m\n\
        OTEL Endpoint: \033[1;36mhttp://localhost:{{ port_signoz_otel }}\033[0m\n"

# Remove SigNoz from the cluster
[group('signoz')]
delete-signoz-agent:
    helm uninstall k8s-infra

# Remove SigNoz from the cluster
[group('signoz')]
delete-signoz: (delete-cluster cluster_signoz)

[group('signoz')]
[private]
apply-signoz-agent cluster:
    @echo "Installing SigNoz OTel Agent"
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
    @bash ./packages/grid/scripts/wait_for.sh service signoz-frontend \
        --namespace platform --context {{ cluster_signoz }} &> /dev/null

    @echo "Setting up SigNoz account"
    @curl --retry 5 --retry-all-errors -X POST \
        -H "Content-Type: application/json" \
        --data '{"email":"admin@localhost","name":"admin","orgName":"openmined","password":"password"}' \
        http://localhost:3301/api/v1/register

    @printf '\nSignoz is running on http://localhost:3301\n\
        Email: \033[1;36madmin@localhost\033[0m\n\
        Password: \033[1;36mpassword\033[0m\n'


# ---------------------------------------------------------------------------------------------------------------------