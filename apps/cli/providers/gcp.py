import subprocess

import click
from PyInquirer import prompt

from ..deploy import base_setup
from ..tf import *
from ..utils import Config, styles
from .provider import *


class GCloud:
    def projects_list(self):
        proc = subprocess.Popen(
            'gcloud projects list --format="value(projectId)"',
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        projects = proc.stdout.read()
        return projects.split()

    def regions_list(self):
        proc = subprocess.Popen(
            'gcloud compute regions list --format="value(NAME)"',
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        regions = proc.stdout.read()
        return regions.split()

    def zones_list(self, region):
        proc = subprocess.Popen(
            f'gcloud compute zones list --filter="REGION:( {region} )" --format="value(NAME)"',
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        zones = proc.stdout.read()
        return zones.split()

    def machines_type(self, zone):
        proc = subprocess.Popen(
            f'gcloud compute machine-types list --filter="ZONE:( {zone} )" --format="value(NAME)"',
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        machines = proc.stdout.read()
        return machines.split()

    def images_type(self):
        proc = subprocess.Popen(
            f'gcloud compute images list --format="value(NAME,PROJECT,FAMILY)"',
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        images = proc.stdout.read().split()
        images = {
            images[i]: (images[i + 1], images[i + 2]) for i in range(0, len(images), 3)
        }
        return images


class GCP(Provider):
    """Google Cloud Provider."""

    def __init__(self, config):
        super().__init__(config)

        self.config.gcp = self.get_gcp_config()

        self.tfscript += terrascript.provider.google(
            project=self.config.gcp.project_id,
            region=self.config.gcp.region,
            zone=self.config.gcp.zone,
        )

        self.update_script()

        click.echo("Initializing GCP Provider")
        TF.init()

        build = self.build()

        if build == 0:
            click.echo("Main Infrastructure has built Successfully!\n\n")

    def build(self) -> bool:
        app = self.config.app.name
        self.firewall = terrascript.resource.google_compute_firewall(
            f"firewall-{app}",
            name=f"firewall-{app}",
            network="default",
            allow={
                "protocol": "tcp",
                "ports": ["80", "443", "5000-5999", "6000-6999", "7000-7999",],
            },
        )
        self.tfscript += self.firewall

        self.pygrid_ip = terrascript.resource.google_compute_address(
            f"pygrid-{app}", name=f"pygrid-{app}",
        )
        self.tfscript += self.pygrid_ip

        self.tfscript += terrascript.output(
            f"pygrid-{app}_ip", value="${" + self.pygrid_ip.address + "}",
        )

        self.update_script()
        return TF.validate()

    def deploy_network(
        self, name: str = "pygridnetwork", apply: bool = True,
    ):
        images = self.config.gcp.images
        image_type = self.config.gcp.image_type
        image = terrascript.data.google_compute_image(
            name + image_type,
            project=images[image_type][0],
            family=images[image_type][1],
        )
        self.tfscript += image

        network = terrascript.resource.google_compute_instance(
            name,
            name=name,
            machine_type=self.config.gcp.machine_type,
            zone=self.config.gcp.zone,
            boot_disk={"initialize_params": {"image": "${" + image.self_link + "}"}},
            network_interface={
                "network": "default",
                "access_config": {"nat_ip": "${" + self.pygrid_ip.address + "}"},
            },
            metadata_startup_script=f"""
                {base_setup}
                \ncd /PyGrid/apps/network
                \npoetry install
                \nnohup ./run.sh --port {self.config.app.port}  --host {self.config.app.host} {'--start_local_db' if self.config.app.start_local_db else ''}
            """,
        )
        self.tfscript += network

        self.update_script()

        return TF.apply()

    def deploy_node(
        self, name: str = "pygridnode", apply: bool = True,
    ):
        images = self.config.gcp.images
        image_type = self.config.gcp.image_type
        image = terrascript.data.google_compute_image(
            name + image_type,
            project=images[image_type][0],
            family=images[image_type][1],
        )
        self.tfscript += image

        network = terrascript.resource.google_compute_instance(
            name,
            name=name,
            machine_type=self.config.gcp.machine_type,
            zone=self.config.gcp.zone,
            boot_disk={"initialize_params": {"image": "${" + image.self_link + "}"}},
            network_interface={"network": "default", "access_config": {},},
            metadata_startup_script=f"""
                {base_setup}
                \ncd /PyGrid/apps/node
                \npoetry install
                \nnohup ./run.sh --id {self.config.app.id} --port {self.config.app.port}  --host {self.config.app.host} --network {self.config.app.network} --num_replicas {self.config.app.num_replicas} {'--start_local_db' if self.config.app.start_local_db else ''}
            """,
        )
        self.tfscript += network

        self.update_script()

        return TF.apply()

    def get_gcp_config(self) -> Config:
        """Getting the configration required for deployment on GCP.

        Returns:
            Config: Simple Config with the user inputs
        """
        gcp = GCloud()

        project_id = prompt(
            [
                {
                    "type": "list",
                    "name": "project_id",
                    "message": "Please select your project_id",
                    "choices": gcp.projects_list(),
                },
            ],
            style=styles.second,
        )["project_id"]

        region = prompt(
            [
                {
                    "type": "list",
                    "name": "region",
                    "message": "Please select your desired GCP region",
                    "default": "us-central1",
                    "choices": gcp.regions_list(),
                },
            ],
            style=styles.second,
        )["region"]

        zone = prompt(
            [
                {
                    "type": "list",
                    "name": "zone",
                    "message": "Please select your desired GCP zone",
                    "choices": gcp.zones_list(region),
                },
            ],
            style=styles.second,
        )["zone"]

        machine_type = prompt(
            [
                {
                    "type": "list",
                    "name": "machine_type",
                    "message": "Please select your desired Machine type",
                    "choices": gcp.machines_type(zone),
                },
            ],
            style=styles.second,
        )["machine_type"]

        images = gcp.images_type()
        image_type = prompt(
            [
                {
                    "type": "list",
                    "name": "image_type",
                    "message": "Please select your desired Machine type",
                    "choices": images.keys(),
                },
            ],
            style=styles.second,
        )["image_type"]

        return Config(
            project_id=project_id,
            region=region,
            zone=zone,
            machine_type=machine_type,
            images=images,
            image_type=image_type,
        )
