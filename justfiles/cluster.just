# ---------------------------------------------------------------------------------------------------------------------

# Launch a Datasite high-side cluster on http://localhost:{{port_high}}
[group('highside')]
start-high: (delete-cluster cluster_high) (create-cluster cluster_high port_high)

# Stop the Datasite high-side cluster
[group('highside')]
delete-high: (delete-cluster cluster_high)

# Deploy Syft to the high-side cluster
[group('highside')]
deploy-high: (deploy-devspace cluster_high ns_default)

# Deploy Syft to the high-side cluster in Hot Reload Mode
[group('highside')]
hotreload-high: (hotreload-devspace cluster_high ns_default)

# Reset Syft DB state in the high-side cluster
[group('highside')]
reset-high: (reset-syft cluster_high ns_default)

# Remove devpsace deployment + namespace from the high-side cluster
[group('highside')]
cleanup-high: (purge-devspace cluster_high ns_default) (delete-ns cluster_high ns_default)

# ---------------------------------------------------------------------------------------------------------------------

# Launch a Datasite low-side cluster on http://localhost:{{port_low}}
[group('lowside')]
start-low: (create-cluster cluster_low port_low)

# Stop the Datasite low-side cluster
[group('lowside')]
delete-low: (delete-cluster cluster_low)

# Deploy Syft to the low-side cluster
[group('lowside')]
deploy-low: (deploy-devspace cluster_low ns_default "-p datasite-low")

# Deploy Syft to the low-side cluster in Hot Reload Mode
[group('lowside')]
hotreload-low: (hotreload-devspace cluster_low ns_default "-p datasite-low")

# Reset Syft DB state in the low-side cluster
[group('lowside')]
reset-low: (reset-syft cluster_low ns_default)

# Remove devpsace deployment + namespace from the low-side cluster
[group('lowside')]
cleanup-low: (purge-devspace cluster_low ns_default) (delete-ns cluster_low ns_default)

# ---------------------------------------------------------------------------------------------------------------------

# Launch a Gateway cluster on http://localhost:{{port_gw}}
[group('gateway')]
start-gw: (create-cluster cluster_gw port_gw)

# Delete the Gateway cluster
[group('gateway')]
delete-gw: (delete-cluster cluster_gw)

# Deploy Syft to the gateway cluster
[group('gateway')]
deploy-gw: (deploy-devspace cluster_gw ns_default "-p gateway")

# Deploy Syft to the gateway cluster in Hot Reload Mode
[group('gateway')]
hotreload-gw: (hotreload-devspace cluster_gw ns_default "-p gateway")

# Reset Syft DB state in the gateway cluster
[group('gateway')]
reset-gw: (reset-syft cluster_gw ns_default)

# Remove devpsace deployment + namespace from the gateway cluster
[group('gateway')]
cleanup-gw: (purge-devspace cluster_gw ns_default) (delete-ns cluster_gw ns_default)

# ---------------------------------------------------------------------------------------------------------------------

# TODO - multi-namespace -> unique k3d ports
# # Launch a multi-agent cluster on http://localhost:{{port_default}}
# [group('shared')]
# start-shared: (create-cluster cluster_default port_default "--agents 2")

# # Stop the multi-agent cluster
# [group('shared')]
# delete-shared: (delete-cluster cluster_default)

# [group('shared')]
# deploy-ns-high: (deploy-devspace cluster_default ns_high)

# [group('shared')]
# delete-ns-high: (delete-ns cluster_default ns_high)

# [group('shared')]
# deploy-ns-low: (deploy-devspace cluster_default ns_low "-p datasite-low")

# [group('shared')]
# delete-ns-low: (delete-ns cluster_default ns_low)

# [group('shared')]
# deploy-ns-gw: (deploy-devspace cluster_default ns_gw "-p gateway")

# [group('shared')]
# delete-ns-gw: (delete-ns cluster_default ns_gw)
