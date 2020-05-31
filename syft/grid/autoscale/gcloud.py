"""To autoscale pygrid workers on Google Cloud Platfrom"""
import json
import IPython
import terrascript
import terrascript.provider
import terrascript.resource
from utils.script import terraform_script
from utils.notebook import terraform_notebook


class GoogleCloud:
    """This class defines automates the spinning up of Google Cloud Instances"""

    def __init__(self, credentials, project_id, region):
        """
        args:
            credentials: Path to the credentials json file
            project_id: project_id of your project in GCP
            region: region of your GCP project
        """
        self.credentials = credentials
        self.project_id = project_id
        self.region = region
        self.config = terrascript.Terrascript()
        self.config += terrascript.provider.google(
            credentials=self.credentials, project=self.project_id, region=self.region
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if IPython.get_ipython():
            terraform_notebook.init()
        else:
            terraform_script.init()

    def compute_instance(self, name, machine_type, zone, image_family, apply=True):
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
            machine_type=machine_type.value,
            zone=zone.value,
            boot_disk={"initialize_params": {"image": image_family.value}},
            network_interface={"network": "default", "access_config": {}},
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def create_cluster(self, name, machine_type, zone, image_family, target_size, apply=True):
        """
        args:
            name: name of the compute instance
            machine_type: the type of machine
            zone: zone of your GCP project
            image_family: image of the OS
            target_size: number of wokers to be created(N workers + 1 master)
        """
        if target_size < 3:
            raise ValueError("The target-size should be equal to or greater than three.")

        self.compute_instance(name, machine_type, zone, image_family, False)

        instance_template = terrascript.resource.google_compute_instance_template(
            "worker-template",
            name=name + "-worker-template",
            machine_type=machine_type.value,
            disk={"source_image": image_family.value},
            network_interface={"network": "default", "access_config": {}},
            lifecycle={"create_before_destroy": True},
        )
        self.config += instance_template

        self.config += terrascript.resource.google_compute_instance_group_manager(
            name,
            name=name,
            version={"instance_template": "${" + instance_template.self_link + "}"},
            base_instance_name=name,
            zone=zone.value,
            target_size=str(target_size),
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def destroy(self):
        """
        args:
        """
        if IPython.get_ipython():
            terraform_notebook.destroy()
        else:
            terraform_script.destroy()
        del self.credentials
