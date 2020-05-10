"""To autoscale pygrid workers on Google Cloud Platfrom"""
import os
import json
import shutil
import subprocess
import terrascript
import terrascript.provider
import terrascript.resource


class GoogleCloud:
    """This class defines automates the spinning up of Google Cloud Instances"""

    def __init__(self, credentials, project_id, region):
        """
        args:
            credentials: Path to the credentials json file
            project_id: project_id of your project in GCP
            region: region of your GCP project
        """
        shutil.copy2(credentials, os.getcwd() + "/credentials.json")
        self.credentials = "credentials.json"
        self.project_id = project_id
        self.region = region
        self.config = terrascript.Terrascript()
        self.config += terrascript.provider.google(
            credentials=self.credentials, project=self.project_id, region=self.region
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)
        subprocess.call("terraform init", shell=True)

    def compute_instance(self, name, machine_type, zone, image_family):
        """
        args:
            name: name of the compute instance
            machine_type: the type of machine
            zone: zone of your GCP project
            image_family: image of the OS
        """
        self.config += terrascript.resource.google_compute_instance(
            name,
            name=name,
            machine_type=machine_type,
            zone=zone,
            boot_disk={"initialize_params": {"image": image_family}},
            network_interface={"network": "default", "access_config": {}},
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)
        # subprocess.call("terraform plan", shell=True)
        subprocess.call("terraform apply", shell=True)

    def destroy(self):
        """
        args:
        """
        subprocess.call("terraform destroy", shell=True)
        os.remove(self.credentials)
