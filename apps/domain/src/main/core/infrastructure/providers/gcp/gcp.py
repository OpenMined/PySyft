import subprocess
import textwrap

import click
from PyInquirer import prompt

from ...tf import generate_cidr_block, var, var_module
from ..provider import *


class GCP(Provider):
    """Google Cloud Provider."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config.root_dir, "gcp")

        self.config = config

        self.tfscript += terrascript.provider.google(
            project=self.config.gcp.project_id,
            region=self.config.gcp.region,
            zone=self.config.gcp.zone,
        )
        click.echo("Initializing GCP Provider")

        self.build()
        self.build_instances()

    def build(self) -> bool:
        app = self.config.app.name
        self.vpc = Module(
            "pygrid-vpc",
            source="terraform-google-modules/network/google",
            project_id=self.config.gcp.project_id,
            network_name=f"pygrid-{app}-vpc",
            routing_mode="GLOBAL",
            auto_create_subnetworks=True,
            subnets=[
                {
                    "subnet_name": "pygrid-subnet",
                    "subnet_ip": "10.10.10.0/24",
                    "subnet_region": self.config.gcp.region,
                }
            ],
            routes=[
                {
                    "name": "egress-internet",
                    "description": "route through IGW to access internet",
                    "destination_range": "0.0.0.0/0",
                    "tags": "egress-inet",
                    "next_hop_internet": "true",
                },
            ],
        )
        self.tfscript += self.vpc

        self.firewall = terrascript.resource.google_compute_firewall(
            f"firewall-{app}",
            name=f"firewall-{app}",
            network="default",
            allow={
                "protocol": "tcp",
                "ports": [
                    "80",
                    "8080",
                    "443",
                    "5000-5999",
                    "6000-6999",
                    "7000-7999",
                ],
            },
        )
        self.tfscript += self.firewall

        self.pygrid_ip = terrascript.resource.google_compute_address(
            f"pygrid-{app}", name=f"pygrid-{app}"
        )
        self.tfscript += self.pygrid_ip

        self.tfscript += terrascript.output(
            f"pygrid-{app}-ip", value=var(self.pygrid_ip.address)
        )

    def build_instances(self):
        name = self.config.app.name
        images = vars(self.config.gcp.images)
        image_type = self.config.gcp.image_type
        # print(images)
        # print(image_type)
        image = terrascript.data.google_compute_image(
            f"{name}-{image_type}",
            project=images[image_type][0],
            family=images[image_type][1],
        )
        self.tfscript += image

        self.instances = []
        for count in range(self.config.app.count):
            app = self.config.apps[count]

            instance = terrascript.resource.google_compute_instance(
                name,
                name=name,
                machine_type=self.config.gcp.machine_type,
                zone=self.config.gcp.zone,
                boot_disk={"initialize_params": {"image": var(image.self_link)}},
                network_interface={
                    "network": "default",
                    "access_config": {"nat_ip": var(self.pygrid_ip.address)},
                },
                metadata_startup_script=self.write_exec_script(app, index=count),
            )

            self.tfscript += instance
            self.instances.append(instance)

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

            nohup ./run.sh --port {app.port}  --start_local_db
        """
        )
        return exec_script
