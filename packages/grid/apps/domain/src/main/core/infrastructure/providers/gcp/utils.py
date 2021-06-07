# stdlib
import subprocess

# third party
from PyInquirer import prompt
import click

# grid relative
from ...utils import Config
from ...utils import styles


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
            f'gcloud compute machine-types list --filter="ZONE:( {zone} )" --format="value(NAME,CPUS,MEMORY_GB)"',
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        machines = proc.stdout.read().split()
        machines = [
            f"Machine Name: {machines[i]} | CPUs: {machines[i + 1]} | Memory: {machines[i + 2]}"
            for i in range(0, len(machines), 3)
        ]
        return machines

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


def get_all_instance_types(zone=None):
    proc = subprocess.Popen(
        f'gcloud compute machine-types list --filter="ZONE:( {zone} )" --format="value(NAME)"',
        shell=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    machines = proc.stdout.read().split()
    # machines = [
    #     f"Machine Name: {machines[i]} | CPUs: {machines[i + 1]} | Memory: {machines[i + 2]}"
    #     for i in range(0, len(machines), 3)
    # ]
    return {"all_instances": machines}


def get_gcp_config() -> Config:
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
            }
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
            }
        ],
        style=styles.second,
    )["region"]

    zone = prompt(
        [
            {
                "type": "list",
                "name": "zone",
                "message": "Please select at least two availability zones. (Not sure? Select the first two)",
                "choices": gcp.zones_list(region),
            }
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
            }
        ],
        style=styles.second,
    )["machine_type"].split()[2]

    images = gcp.images_type()
    image_type = prompt(
        [
            {
                "type": "list",
                "name": "image_type",
                "message": "Please select your desired Machine type",
                "choices": images.keys(),
            }
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
