import subprocess

import click
from PyInquirer import prompt

from ...utils import Config, styles


class AZ:
    def locations_list(self):
        proc = subprocess.Popen(
            "az account list-locations --query '[].{DisplayName:displayName}' --output table",
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        locations = proc.stdout.read()
        return locations.split("\n")[2:]


def get_azure_config() -> Config:
    """Getting the configration required for deployment on AZURE.

    Returns:
        Config: Simple Config with the user inputs
    """

    az = AZ()

    subscription_id = prompt(
        [
            {
                "type": "input",
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
                "type": "input",
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
                "type": "input",
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
                "type": "input",
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

    return Config(
        location=location,
        subscription_id=subscription_id,
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
    )
