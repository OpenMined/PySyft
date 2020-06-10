"""To test the implementation of gcloud.py"""
import syft.grid.autoscale.gcloud as gcloud
import syft.grid.autoscale.utils.gcloud_configurations as configs


NEW = gcloud.GoogleCloud(
    credentials="/usr/terraform.json", project_id="project", region=configs.Region.us_central1,
)

NEW.create_cluster(
    name="new-12345",
    machine_type=configs.MachineType.f1_micro,
    zone=configs.Zone.us_central1_a,
    image_family=configs.ImageFamily.ubuntu_2004_lts,
    target_size=3,
)

NEW.destroy()
