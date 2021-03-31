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
        self.tfscript += terrascript.terraform(
            required_providers={
                "azurerm": {"source": "hashicorp/azurerm", "version": "=2.46.0"}
            }
        )
        self.tfscript += azurerm(
            features={},
            subscription_id=self.config.azure.subscription_id,
            client_id=self.config.azure.client_id,
            client_secret=self.config.azure.client_secret,
            tenant_id=self.config.azure.tenant_id,
        )

        self.build_resource_group()
        self.build_security_groups()
        self.build_network()

        self.build_instances()
        # self.build_load_balancer()

    def build_resource_group(self):
        self.resource_group = azurerm_resource_group(
            f"pygrid_resource_group",
            name=f"pygrid_resource_group",
            location=self.config.azure.location,
        )
        self.tfscript += self.resource_group

    def build_network(self):
        self.network = vnet(
            f"pygrid-network",
            source="Azure/vnet/azurerm",
            resource_group_name=self.resource_group.name,
            address_space=["10.0.0.0/16"],
            subnet_prefixes=["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"],
            subnet_names=["subnet1", "subnet2", "subnet3"],
            nsg_ids={
                "subnet1": var(self.network_security_group.id),
                "subnet2": var(self.network_security_group.id),
                "subnet3": var(self.network_security_group.id),
            },
            tags={
                "name": "pygrid-virtual-network",
                "environment": "dev",
            },
            depends_on=["azurerm_resource_group.pygrid_resource_group"],
        )
        self.tfscript += self.network

    def build_security_groups(self):
        self.network_security_group = azurerm_network_security_group(
            f"pygrid_network_security_group",
            name=f"pygrid_network_security_group",
            resource_group_name=var(
                "azurerm_resource_group.pygrid_resource_group.name"
            ),
            location=self.resource_group.location,
            tags={
                "name": "pygrid-security-groups",
                "environment": "dev",
            },
        )
        self.tfscript += self.network_security_group

        security_rules = [
            ("HTTPS", 100, 443, 443),
            ("HTTP", 101, 80, 80),
            ("PyGrid_Domains", 102, 5000, 5999),
            ("PyGrid_Workers", 103, 6000, 6999),
            ("PyGrid_Networks", 104, 7000, 7999),
            ("SSH", 105, 22, 22),
        ]
        for name, priority, source_port, dest_port in security_rules:
            self.tfscript += azurerm_network_security_rule(
                name,
                name=name,
                priority=priority,
                direction="Inbound",
                access="Allow",
                protocol="Tcp",
                source_port_range=source_port,
                destination_port_range=dest_port,
                source_address_prefix="*",
                destination_address_prefix="*",
                resource_group_name=var(
                    "azurerm_resource_group.pygrid_resource_group.name"
                ),
                network_security_group_name=var(
                    "azurerm_network_security_group.pygrid_network_security_group.name"
                ),
            )

    def build_instances(self):
        name = self.config.app.name

        self.instances = []
        for count in range(self.config.app.count):
            app = self.config.apps[count]

            instance = linuxservers(
                f"pygrid-{name}-instance-{count}",
                source="Azure/compute/azurerm",
                resource_group_name=self.resource_group.name,
                vm_os_simple="UbuntuServer",
                public_ip_dns=[f"pygrid-{name}-instance-{count}"],
                vnet_subnet_id=var_module(self.network, "vnet_subnets[0]"),
                custom_data=self.write_exec_script(app, index=count),
                admin_username="pygriduser",
                admin_password="pswd123!",
                depends_on=["azurerm_resource_group.pygrid_resource_group"],
            )

            self.tfscript += instance
            self.tfscript += terrascript.Output(
                f"instance_{count}_endpoint",
                value=var_module(instance, "public_ip_address"),
                description=f"The public IP address of #{count} instance.",
            )
            self.instances.append(instance)

    def build_load_balancer(self):
        self.load_balancer = mylb(
            f"pygrid_load_balancer",
            source="Azure/loadbalancer/azurerm",
            resource_group_name=self.resource_group.name,
            prefix="terraform-lb",
            depends_on=["azurerm_resource_group.pygrid_resource_group"],
            frontend_subnet_id=var_module(self.network, "vnet_subnets[0]"),
            remote_port={"ssh": ["Tcp", "22"]},
            lb_port={
                "http": ["80", "Tcp", "80"],
                "https": ["443", "Tcp", "443"],
            },
            lb_probe={
                "http": ["Tcp", "80", ""],
                "http2": ["Http", "1443", "/"],
            },
        )
        self.tfscript += self.load_balancer
        self.tfscript += terrascript.Output(
            f"load_balancer_dns",
            value=var_module(self.load_balancer, "azurerm_public_ip_address"),
            description="The DNS name of the ELB.",
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
