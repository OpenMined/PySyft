# ---------------------------------------------------------------------------------------------------------------------

cluster_default := "syft-dev"
cluster_high := "syft-high"
cluster_low := "syft-low"
cluster_gw := "syft-gw"
cluster_signoz := "signoz"
ns_default := "syft"
ns_high := "high"
ns_low := "low"
ns_gw := "gw"

# ---------------------------------------------------------------------------------------------------------------------

port_default := "8080"
port_high := port_default
port_low := "8081"
port_gw := "8082"
port_signoz_ui := "3301"
port_signoz_otel := "4317"
port_registry := "5800"

registry_url := "k3d-registry.localhost:" + port_registry
signoz_otel_url := "http://host.k3d.internal:" + port_signoz_otel

# ---------------------------------------------------------------------------------------------------------------------

# devspace profiles (comma-separated)
profiles := ""

# enable tracing by adding "tracing" profile in devspace
tracing := "true"

_g_profiles := if tracing == "true" { profiles + ",tracing" } else { profiles }

# ---------------------------------------------------------------------------------------------------------------------

# this might break if you have alias python = python3 or either of the executable not pointing to the correct one
# just fix your system instead of making of fixing this
python_path := `which python || which python3`

# ---------------------------------------------------------------------------------------------------------------------
