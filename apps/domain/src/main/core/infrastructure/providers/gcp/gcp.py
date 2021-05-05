# stdlib
import subprocess
import textwrap

# third party
from PyInquirer import prompt
import click

# grid relative
from ...tf import generate_cidr_block
from ...tf import var
from ...tf import var_module
from ..provider import *


class GCP(Provider):
    """Google Cloud Provider."""

    def __init__(self, config: SimpleNamespace) -> None:
        config.root_dir = os.path.join(
            str(Path.home()),
            ".pygrid",
            "apps",
            str(config.provider),
            str(config.app.name),
            str(vars(config.app).get("id", "")),
        )
        super().__init__(config.root_dir, "gcp")

        self.config = config
        self.name = self.config.app.name + str(vars(config.app).get("id", ""))

        self.tfscript += terrascript.provider.google(
            project=self.config.gcp.project_id,
            region=self.config.gcp.region,
            zone=self.config.gcp.zone,
        )
        click.echo("Initializing GCP Provider")

        self.worker = config.app.name == "worker"

        if self.worker:
            # self.build()
            self.build_ip_address()
            self.build_instances()
        else:
            self.build()
            self.build_ip_address()
            self.build_instances()

    def build(self) -> bool:
        app = self.name
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

    def build_ip_address(self):
        app = self.name
        self.pygrid_ip = terrascript.resource.google_compute_address(
            f"pygrid-{app}", name=f"pygrid-{app}"
        )
        self.tfscript += self.pygrid_ip

        self.tfscript += terrascript.output(
            f"instance_0_endpoint", value=var(self.pygrid_ip.address)
        )

    def build_instances(self):
        name = self.name

        self.instances = []
        for count in range(self.config.app.count):
            app = self.config.apps[count]

            instance = terrascript.resource.google_compute_instance(
                name,
                name=name,
                machine_type=self.config.gcp.machine_type,
                zone=self.config.gcp.zone,
                boot_disk={
                    "initialize_params": {"image": "ubuntu-os-cloud/ubuntu-1804-lts"}
                },
                network_interface={
                    "network": "default",
                    "access_config": {"nat_ip": var(self.pygrid_ip.address)},
                },
                metadata_startup_script=self.write_domain_exec_script(app, index=count)
                if not self.worker
                else self.write_worker_exec_script(app),
            )

            self.tfscript += instance
            self.instances.append(instance)

    def write_domain_exec_script(self, app, index=0):
        ##TODO(amr): remove `git checkout pygrid_0.3.0` after merge
        branch = "dev"

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
            sudo apt-get install zip unzip -y
            sudo apt-get install python3-dev -y
            sudo apt-get install libevent-dev -y
            sudo apt-get install gcc -y

            curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
            sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main" -y
            sudo apt-get update -y && sudo apt-get install terraform -y

            echo "Setting environment variables"
            export CLOUD_PROVIDER={self.config.provider}
            export MEMORY_STORE=True
            echo "Exporting GCP Configs"
            export project_id={self.config.gcp.project_id},
            export REGION={self.config.gcp.region},
            export zone={self.config.gcp.zone},
            export machine_type={self.config.gcp.machine_type},

            echo 'Cloning PyGrid'
            git clone https://github.com/OpenMined/PyGrid && cd /PyGrid/
            git checkout {branch}

            cd /PyGrid/apps/{self.config.app.name}

            echo 'Installing {self.config.app.name} Dependencies'
            poetry install

            ## TODO(amr): remove this after poetry updates
            pip install pymysql

            nohup ./run.sh --port {app.port}  --host 0.0.0.0 --start_local_db
            """
        )
        return exec_script

    def write_worker_exec_script(self, app):
        branch = "dev"
        exec_script = "#!/bin/bash\n"
        exec_script += textwrap.dedent(
            f"""
            exec &> logs.out
            sudo apt update -y

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
            sudo apt-get install zip unzip -y
            sudo apt-get install python3-dev -y
            sudo apt-get install libevent-dev -y
            sudo apt-get install gcc -y

            echo 'Cloning PyGrid'
            git clone https://github.com/OpenMined/PyGrid && cd /PyGrid/
            git checkout {branch}

            cd /PyGrid/apps/worker
            echo 'Installing worker Dependencies'
            poetry install
            nohup ./run.sh --port {app.port}  --host 0.0.0.0
            """
        )
        return exec_script
