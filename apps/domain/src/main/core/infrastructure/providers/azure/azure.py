import subprocess
import textwrap

import click
from PyInquirer import prompt

from ...tf import generate_cidr_block, var, var_module
from ..provider import *
from .azure_ts import *


class AZURE(Provider):
    """Azure Cloud Provider."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config.root_dir, "azure")
        self.config = config

        ##TODO(amr): terrascript does not support azurem right now
        # self.tfscript += terrascript.terraform(backend=terrascript.backend("azurerm"))
        self.tfscript += azurerm(
            features={},
            subscription_id=self.config.azure.subscription_id,
            client_id=self.config.azure.client_id,
            client_secret=self.config.azure.client_secret,
            tenant_id=self.config.azure.tenant_id,
        )

        self.build_resource_group()
        self.build_virtual_network()
        self.build_subnets()
        self.build_public_ip()

        self.build_security_groups()
        self.build_network_interface()

        self.build_instances()
        self.build_load_balancer()

    def build_resource_group(self):
        self.resource_group = azurerm_resource_group(
            f"pygrid_resource_group",
            name=f"pygrid_resource_group",
            location=self.config.azure.location,
        )
        self.tfscript += self.resource_group

    def build_virtual_network(self):
        self.virtual_network = azurerm_virtual_network(
            f"pygrid_virtual_network",
            name=f"pygrid_virtual_network",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            address_space=["10.0.0.0/16"],
            tags={
                "name": "pygrid-virtual-network",
                "environment": "dev",
            },
        )
        self.tfscript += self.virtual_network

    def build_subnets(self):
        self.azurerm_subnet = azurerm_subnet(
            f"pygrid_subnet",
            name=f"pygrid_subnet",
            resource_group_name=self.resource_group.name,
            virtual_network_name=self.virtual_network.name,
            address_prefixes=["10.0.2.0/24"],
        )
        self.tfscript += self.azurerm_subnet

        self.availability_set = azurerm_availability_set(
            f"pygrid_availability_set",
            name=f"pygrid_availability_set",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            platform_fault_domain_count=self.config.app.count,
            platform_update_domain_count=self.config.app.count,
            managed=True,
        )
        self.tfscript += self.availability_set

    def build_public_ip(self):
        self.public_ip = azurerm_public_ip(
            f"pygrid_public_ip",
            name=f"pygrid_public_ip",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            allocation_method="Dynamic",
            tags={
                "name": "pygrid-public-ip",
                "environment": "dev",
            },
        )
        self.tfscript += self.public_ip

    def build_security_groups(self):
        self.network_security_group = azurerm_network_security_group(
            f"pygrid_network_security_group",
            name=f"pygrid_network_security_group",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            # security_rule=[
            #     {
            #         "name": "HTTPS",
            #         "priority": 100,
            #         "direction": "Inbound",
            #         "access": "Allow",
            #         "protocol": "Tcp",
            #         "source_port_range": "443",
            #         "destination_port_range": "443",
            #         "source_address_prefix": "*",
            #         "destination_address_prefix": "*",
            #     },
            #     {
            #         "name": "HTTP",
            #         "priority": 101,
            #         "direction": "Inbound",
            #         "access": "Allow",
            #         "protocol": "Tcp",
            #         "source_port_range": "80",
            #         "destination_port_range": "80",
            #         "source_address_prefix": "*",
            #         "destination_address_prefix": "*",
            #     },
            #     {
            #         "name": "PyGrid Domains",
            #         "priority": 102,
            #         "direction": "Inbound",
            #         "access": "Allow",
            #         "protocol": "Tcp",
            #         "source_port_range": "5000",
            #         "destination_port_range": "5999",
            #         "source_address_prefix": "*",
            #         "destination_address_prefix": "*",
            #     },
            #     {
            #         "name": "PyGrid Workers",
            #         "priority": 103,
            #         "direction": "Inbound",
            #         "access": "Allow",
            #         "protocol": "Tcp",
            #         "source_port_range": "6000",
            #         "destination_port_range": "6999",
            #         "source_address_prefix": "*",
            #         "destination_address_prefix": "*",
            #     },
            #     {
            #         "name": "PyGrid Networks",
            #         "priority": 104,
            #         "direction": "Inbound",
            #         "access": "Allow",
            #         "protocol": "Tcp",
            #         "source_port_range": "7000",
            #         "destination_port_range": "7999",
            #         "source_address_prefix": "*",
            #         "destination_address_prefix": "*",
            #     },
            # ],
            tags={
                "name": "pygrid-security-groups",
                "environment": "dev",
            },
        )
        self.tfscript += self.network_security_group

    def build_network_interface(self):

        self.network_interface = azurerm_network_interface(
            f"pygrid_network_interface",
            name=f"pygrid_network_interface",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            ip_configuration={
                "name": "internal",
                "subnet_id": var(self.azurerm_subnet.id),
                "private_ip_address_allocation": "Dynamic",
            },
            tags={
                "name": "pygrid-network-interface",
                "environment": "dev",
            },
        )
        self.tfscript += self.network_interface

        self.ni_sec_association = azurerm_network_interface_security_group_association(
            f"network_interface_security_group_association",
            network_interface_id=var(self.network_interface.id),
            network_security_group_id=var(self.network_security_group.id),
        )
        self.tfscript += self.ni_sec_association

    def build_instances(self):
        name = self.config.app.name

        self.instances = []
        for count in range(self.config.app.count):
            app = self.config.apps[count]

            instance = azurerm_virtual_machine(
                name,
                name=name,
                resource_group_name=self.resource_group.name,
                location=self.resource_group.location,
                network_interface_ids=[var(self.network_interface.id)],
                availability_set_id=var(self.availability_set.id),
                vm_size="Standard_DS1_v2",  # TODO: get this config from user
                # TODO: get config from user
                storage_image_reference={
                    "publisher": "Canonical",
                    "offer": "UbuntuServer",
                    "sku": "18.04-LTS",
                    "version": "latest",
                },
                storage_os_disk={
                    "name": "OSDisk",
                    "caching": "ReadWrite",
                    "create_option": "FromImage",
                    "managed_disk_type": "Standard_LRS",
                },
                os_profile={
                    "computer_name": "hostname",
                    "admin_username": "openmined",
                    "admin_password": "pswd123!",
                    "custom_data": self.write_exec_script(app, index=count),
                },
                os_profile_linux_config={"disable_password_authentication": False},
                delete_os_disk_on_termination=True,
                delete_data_disks_on_termination=True,
            )

            self.tfscript += instance
            self.instances.append(instance)

    def build_load_balancer(self):
        self.load_balancer = azurerm_lb(
            f"pygrid_load_balancer",
            name=f"pygrid_load_balancer",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            frontend_ip_configuration={
                "name": "lb_public_ip",
                "public_ip_address_id": var(self.public_ip.id),
            },
        )
        self.tfscript += self.load_balancer

        self.lb_backend_address = azurerm_lb_backend_address_pool(
            f"pygrid_lb_backend_address",
            name=f"pygrid_lb_backend_address",
            loadbalancer_id=var(self.load_balancer.id),
        )
        self.tfscript += self.lb_backend_address

    def build_database(self):
        """https://registry.terraform.io/providers/hashicorp/azurerm/latest/doc
        s/resources/sql_database."""
        self.sql_server = azurerm_sql_server(
            f"pygrid_sql_server",
            name=f"pygrid_sql_server",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            version="12.0",
            administrator_login=self.config.credentials.db.username,
            administrator_login_password=self.config.credentials.db.password,
        )
        self.tfscript += self.sql_server

        self.storage_account = azurerm_storage_account(
            f"pygrid_storage_account",
            name=f"pygrid_storage_account",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            account_tier="Standard",
            account_replication_type="LRS",
        )
        self.tfscript += self.storage_account

        self.database = azurerm_sql_database(
            f"pygrid-{self.config.app.name}-database",
            name=f"pygrid-{self.config.app.name}-database",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            server_name=self.sql_server.name,
            extended_auditing_policy={
                "storage_endpoint": self.storage_account.primary_blob_endpoint,
                "storage_account_access_key": self.storage_account.primary_access_key,
                "storage_account_access_key_is_secondary": True,
                "retention_in_days": 6,
            },
        )

    def write_exec_script(self, app, index=0):
        ##TODO(amr): remove `git checkout pygrid_0.3.0` after merge

        # exec_script = "#cloud-boothook\n#!/bin/bash\n"
        exec_script = "#!/bin/bash\n"
        exec_script += textwrap.dedent(
            f"""
            ## For debugging
            # redirect stdout/stderr to a file
            exec &> logs.out

            echo 'Simple Web Server for testing the deployment'
            sudo apt update -y
            sudo apt install apache2 -y
            sudo systemctl start apache2
            echo '<h1>OpenMined {self.config.app.name} Server ({index}) Deployed via Terraform</h1>' | sudo tee /var/www/html/index.html

            echo 'Setup Miniconda environment'
            sudo wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
            sudo bash miniconda.sh -b -p miniconda
            sudo rm miniconda.sh
            export PATH=/miniconda/bin:$PATH > ~/.bashrc
            conda init bash
            source ~/.bashrc
            conda create -y -n pygrid python=3.7
            conda activate pygrid

            echo 'Install poetry...'
            pip install poetry

            echo 'Install GCC'
            sudo apt-get install python3-dev -y
            sudo apt-get install libevent-dev -y
            sudo apt-get install gcc -y

            echo 'Cloning PyGrid'
            git clone https://github.com/OpenMined/PyGrid && cd /PyGrid/
            git checkout pygrid_0.4.0

            cd /PyGrid/apps/{self.config.app.name}

            echo 'Installing {self.config.app.name} Dependencies'
            poetry install

            ## TODO(amr): remove this after poetry updates
            pip install pymysql

            nohup ./run.sh --port {app.port} --start_local_db
        """
        )
        return exec_script
