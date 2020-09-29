import subprocess

import click
from PyInquirer import prompt

from ..deploy import base_setup
from ..tf import *
from ..utils import Config, styles
from .provider import *


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


class AZURE(Provider):
    """Azure Cloud Provider."""

    def __init__(self, config):
        super().__init__(config)

        self.config.azure = self.get_azure_config()

        self.tfscript += terrascript.provider.azurerm()

        self.update_script()

        click.echo("Initializing Azure Provider")
        TF.init()

        build = self.build()

        if build == 0:
            click.echo("Main Infrastructure has built Successfully!\n\n")

    def build(self) -> bool:
        self.resource_group = terrascript.resource.azurerm_resource_group(
            "resource_group", name="resource_group", location=self.azure.location,
        )
        self.tfscript += self.resource_group

        self.virtual_network = terrascript.resource.azurerm_virtual_network(
            "virtual_network",
            name="virtual_network",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            address_space=[self.azure.address_space],
        )
        self.tfscript += self.virtual_network

        self.internal_subnet = terrascript.resource.azurerm_subnet(
            "internal_subnet",
            name="internal_subnet",
            resource_group_name=self.resource_group.name,
            virtual_network_name=self.virtual_network.name,
            address_prefix=self.azure.address_prefix,
        )
        self.tfscript += self.internal_subnet

        self.network_interface = terrascript.resource.azurerm_network_interface(
            "network_interface",
            name="network_interface",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            ip_configuration={
                "name": "ip_configuration",
                "subnet_id": self.internal_subnet.id,
                "private_ip_address_allocation": "Dynamic",
            },
        )
        self.tfscript += self.network_interface

        self.update_script()
        return TF.validate()

    def deploy_network(
        self, name: str = "pygridmetwork", apply: bool = True,
    ):
        virtual_machine = terrascript.resource.azurerm_virtual_machine(
            name,
            name=name,
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            network_interface_ids=[self.network_interface.id],
            vm_size="Standard_DS1_v2",  # TODO: get this config from user
            # TODO: get config from user
            storage_image_reference={
                "publisher": "Canonical",
                "offer": "UbuntuServer",
                "sku": "16.04-LTS",
                "version": "latest",
            },
            storage_os_disk={
                "name": "myosdisk1",
                "caching": "ReadWrite",
                "create_option": "FromImage",
                "managed_disk_type": "Standard_LRS",
            },
            os_profile={
                "computer_name": "hostname",
                "admin_username": "testadmin",
                "admin_password": "Password1234!",
            },
            os_profile_linux_config={"disable_password_authentication": False,},
            custom_data=f"""
                {base_setup}
                \ncd /PyGrid/apps/network
                \npoetry install
                \nnohup ./run.sh --port {self.config.app.port}  --host {self.config.app.host} {'--start_local_db' if self.config.app.start_local_db else ''}
            """,
        )

        self.tfscript += network

        self.update_script()

    def deploy_node(
        self, apply: bool = True,
    ):
        virtual_machine = terrascript.resource.azurerm_virtual_machine(
            name,
            name=name,
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            network_interface_ids=[self.network_interface.id],
            vm_size="Standard_DS1_v2",  # TODO: get this config from user
            # TODO: get config from user
            storage_image_reference={
                "publisher": "Canonical",
                "offer": "UbuntuServer",
                "sku": "16.04-LTS",
                "version": "latest",
            },
            storage_os_disk={
                "name": "myosdisk1",
                "caching": "ReadWrite",
                "create_option": "FromImage",
                "managed_disk_type": "Standard_LRS",
            },
            os_profile={
                "computer_name": "hostname",
                "admin_username": "testadmin",
                "admin_password": "Password1234!",
            },
            os_profile_linux_config={"disable_password_authentication": False,},
            custom_data=f"""
                {base_setup}
                \ncd /PyGrid/apps/node
                \npoetry install
                \nnohup ./run.sh --id {self.config.app.id} --port {self.config.app.port}  --host {self.config.app.host} --network {self.config.app.network} --num_replicas {self.config.app.num_replicas} {'--start_local_db' if self.config.app.start_local_db else ''}
            """,
        )

        self.tfscript += network

        self.update_script()

    def get_azure_config(self) -> Config:
        """Getting the configration required for deployment on AZURE.

        Returns:
            Config: Simple Config with the user inputs
        """

        az = AZ()

        location = prompt(
            [
                {
                    "type": "list",
                    "name": "location",
                    "message": "Please select your desired location",
                    "choices": az.locations_list(),
                },
            ],
            style=styles.second,
        )["location"]

        address_space = prompt(
            [
                {
                    "type": "input",
                    "name": "address_space",
                    "message": "Please provide your VPC address_space",
                    "default": "10.0.0.0/16",
                },
            ],
            style=styles.second,
        )["address_space"]

        address_prefix = prompt(
            [
                {
                    "type": "input",
                    "name": "address_prefix",
                    "message": "Please provide subnet address_prefix",
                    "default": "10.0.0.0/24",
                },
            ],
            style=styles.second,
        )["address_prefix"]

        return Config(
            location=location,
            address_space=address_space,
            address_prefix=address_prefix,
        )
