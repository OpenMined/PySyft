"""To test the implementation of gcloud.py"""
import gcloud


NEW = gcloud.GoogleCloud(
    credentials="GCP Login/terraf.json", project_id="terraform", region="us-central1"
)

NEW.compute_instance(
    name="new-12345", machine_type="f1-micro", zone="us-central1-a", image_family="debian-9",
)

NEW.destroy()
