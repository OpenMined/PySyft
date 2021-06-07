# stdlib
import json
import subprocess

# third party
from PyInquirer import prompt
import click

# grid relative
from ...utils import Config
from ...utils import styles


class AZ:
    def locations_list(self):
        proc = subprocess.Popen(
            "az account list-locations",
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        locations = json.loads(proc.stdout.read())
        return [location["name"] for location in locations]


def get_all_instance_types(location=None):
    proc = subprocess.Popen(
        f"az vm list-sizes --location {location}",
        shell=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    machines = json.loads(proc.stdout.read())
    all_instances = {
        "all_instances": [
            f"Name: {machine['name']} | CPUs: {machine['numberOfCores']} | Mem: {int(machine['memoryInMb']/1024)} "
            for machine in machines
        ]
    }
    return all_instances


def get_azure_config() -> Config:
    """Getting the configration required for deployment on AZURE.

    Returns:
        Config: Simple Config with the user inputs
    """

    az = AZ()

    subscription_id = prompt(
        [
            {
                "type": "password",
                "name": "subscription_id",
                "message": "Please provide your subscription_id",
                "default": "00000000-0000-0000-0000-000000000000",
            }
        ],
        style=styles.second,
    )["subscription_id"]

    client_id = prompt(
        [
            {
                "type": "password",
                "name": "client_id",
                "message": "Please provide your client_id",
                "default": "00000000-0000-0000-0000-000000000000",
            }
        ],
        style=styles.second,
    )["client_id"]

    client_secret = prompt(
        [
            {
                "type": "password",
                "name": "client_secret",
                "message": "Please provide your client_secret",
                "default": "XXXX-XXXX-XXX-XXX-XXX",
            }
        ],
        style=styles.second,
    )["client_secret"]

    tenant_id = prompt(
        [
            {
                "type": "password",
                "name": "tenant_id",
                "message": "Please provide your tenant_id",
                "default": "00000000-0000-0000-0000-000000000000",
            }
        ],
        style=styles.second,
    )["tenant_id"]

    location = prompt(
        [
            {
                "type": "list",
                "name": "location",
                "message": "Please select your desired location",
                "choices": az.locations_list(),
            }
        ],
        style=styles.second,
    )["location"]

    vm_size = prompt(
        [
            {
                "type": "list",
                "name": "VMSize",
                "message": "Please select your desired VM Size",
                "choices": get_all_instance_types(location=location)["all_instances"],
            }
        ],
        style=styles.second,
    )["VMSize"].split(" ")[1]

    return Config(
        location=location,
        subscription_id=subscription_id,
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        vm_size=vm_size,
    )
