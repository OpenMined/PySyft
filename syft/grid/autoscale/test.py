"""To test the implementation of gcloud.py"""
from syft.grid.autoscale import gcloud


NEW = gcloud.GoogleCloud(
    credentials="/usr/terraform.json", project_id="project", region="us-central1",
)

NEW.create_cluster(
    name="new-12345",
    machine_type="f1-micro",
    zone="us-central1-a",
    image_family="debian-9",
    target_size=3,
)

NEW.destroy()
