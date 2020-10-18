"""To test the implementation of gcloud.py"""
import syft.grid.autoscale.gcloud as gcloud
import syft.grid.autoscale.utils.gcloud_configurations as configs


NEW = gcloud.GoogleCloud(
    credentials="/usr/terraform.json",
    project_id="project",
    region=configs.Region.us_central1,
)

# create a grid network instance and then add nodes to it one by one
NEW.create_gridnetwork(
    name="new-network",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
)

# add node to created grid network
NEW.create_gridnode(
    name="new-node",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
    gridnetwork_name="new-network",
)

NEW.destroy()

# to create a cluster we first need to reserve an external ip
NEW.reserve_ip("grid")

# pass the name of the reserved ip to the cluster
c1 = NEW.create_cluster(
    name="new-12345",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
    reserve_ip_name="grid",
    target_size=3,
    eviction_policy="delete",
)

c1.sweep()

NEW.destroy()
