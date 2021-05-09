# stdlib
import textwrap

# grid relative
from .aws import *


class AWS_Serverfull(AWS):
    def __init__(self, config: SimpleNamespace) -> None:
        """
        credentials (dict) : Contains AWS credentials
        """

        self.worker = config.app.name == "worker"

        if self.worker:
            config.root_dir = os.path.join(
                "/home/ubuntu/.pygrid/apps/aws/workers/", str(config.app.id)
            )
            super().__init__(config)

            self.vpc = Config(id=os.environ["VPC_ID"])
            public_subnet_ids = str(os.environ["PUBLIC_SUBNET_ID"]).split(",")
            private_subnet_ids = str(os.environ["PRIVATE_SUBNET_ID"]).split(",")
            self.subnets = [
                (Config(id=private), Config(id=public))
                for private, public in zip(private_subnet_ids, public_subnet_ids)
            ]
            self.build_security_group()
            self.build_instances()

        else:  # Deploy a VPC and domain/network
            config.root_dir = os.path.join(
                str(Path.home()), ".pygrid", "apps", str(config.app.name)
            )
            super().__init__(config)

            # Order matters
            self.build_vpc()
            self.build_igw()
            self.build_public_rt()
            self.build_subnets()

            self.build_security_group()
            self.build_database()

            self.build_instances()
            self.build_load_balancer()

        self.output()

    def output(self):
        for count in range(self.config.app.count):
            self.tfscript += terrascript.Output(
                f"instance_{count}_endpoint",
                value=var_module(self.instances[count], "public_ip"),
                description=f"The public IP address of #{count} instance.",
            )

        if hasattr(self, "load_balancer"):
            self.tfscript += terrascript.Output(
                "load_balancer_dns",
                value=var_module(self.load_balancer, "elb_dns_name"),
                description="The DNS name of the ELB.",
            )

    def build_security_group(self):
        # ----- Security Group ------#
        sg_name = f"pygrid-{self.config.app.name}-" + (
            f"{str(self.config.app.id)}-sg" if self.worker else "sg"
        )
        self.security_group = resource.aws_security_group(
            "security_group",
            name=sg_name,
            vpc_id=self.vpc.id if self.worker else var(self.vpc.id),
            ingress=[
                {
                    "description": "HTTPS",
                    "from_port": 443,
                    "to_port": 443,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "HTTP",
                    "from_port": 80,
                    "to_port": 80,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "PyGrid Domains",
                    "from_port": 5000,
                    "to_port": 5999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "PyGrid Workers",
                    "from_port": 6000,
                    "to_port": 6999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "PyGrid Networks",
                    "from_port": 7000,
                    "to_port": 7999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
            ],
            egress=[
                {
                    "description": "Egress Connection",
                    "from_port": 0,
                    "to_port": 0,
                    "protocol": "-1",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                }
            ],
            tags={"Name": sg_name},
        )
        self.tfscript += self.security_group

    def build_instances(self):
        self.ami = terrascript.data.aws_ami(
            "ubuntu",
            most_recent=True,
            filter=[
                {
                    "name": "name",
                    "values": [
                        "ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"
                    ],
                },
                {"name": "virtualization-type", "values": ["hvm"]},
            ],
            owners=["099720109477"],
        )
        self.tfscript += self.ami

        self.instances = []
        kwargs = {}
        for count in range(self.config.app.count):
            app = self.config.apps[count]

            if self.worker:
                instance_name = f"pygrid-worker-{str(self.config.app.id)}"
                user_data = self.write_worker_exec_script(app)
                subnet_ids = [public_subnet.id for _, public_subnet in self.subnets]
            else:
                instance_name = f"pygrid-{self.config.app.name}-instance-{count}"
                user_data = self.write_domain_exec_script(app, index=count)
                subnet_ids = [
                    var(public_subnet.id) for _, public_subnet in self.subnets
                ]

            instance = Module(
                f"pygrid-instance-{count}",
                name=instance_name,
                instance_count=1,
                source="terraform-aws-modules/ec2-instance/aws",
                ami=var(self.ami.id),
                instance_type=self.config.vpc.instance_type.InstanceType,
                associate_public_ip_address=True,
                monitoring=True,
                vpc_security_group_ids=[var(self.security_group.id)],
                subnet_ids=subnet_ids,
                user_data=user_data,
                tags={"Name": instance_name},
            )

            self.tfscript += instance
            self.instances.append(instance)

    def build_load_balancer(self):
        self.load_balancer = Module(
            "pygrid_load_balancer",
            source="terraform-aws-modules/elb/aws",
            name=f"pygrid-{self.config.app.name}-load-balancer",
            subnets=[var(public_subnet.id) for _, public_subnet in self.subnets],
            security_groups=[var(self.security_group.id)],
            number_of_instances=self.config.app.count,
            instances=[var_module(instance, f"id[0]") for instance in self.instances],
            listener=[
                {
                    "instance_port": "80",
                    "instance_protocol": "HTTP",
                    "lb_port": "80",
                    "lb_protocol": "HTTP",
                },
                {
                    "instance_port": "8080",
                    "instance_protocol": "http",
                    "lb_port": "8080",
                    "lb_protocol": "http",
                },
            ],
            health_check={
                "target": "HTTP:80/",
                "interval": 30,
                "healthy_threshold": 2,
                "unhealthy_threshold": 2,
                "timeout": 5,
            },
            tags={"Name": f"pygrid-{self.config.app.name}-load-balancer"},
        )
        self.tfscript += self.load_balancer

    def write_domain_exec_script(self, app, index=0):
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
            export DATABASE_URL={self.database.engine}+pymysql://{self.database.username}:{self.database.password}@{var(self.database.endpoint)}/{self.database.name}

            export MEMORY_STORE=True
            # export DATABASE_URL="sqlite:///pygrid.db"
            export CLOUD_PROVIDER={self.config.provider}
            export REGION={self.config.vpc.region}
            export VPC_ID={var(self.vpc.id)}
            export PUBLIC_SUBNET_ID={','.join([var(public_subnet.id) for _, public_subnet in self.subnets])}
            export PRIVATE_SUBNET_ID={','.join([var(private_subnet.id) for private_subnet, _ in self.subnets])}

            echo "Writing cloud credentials file"
            export AWS_ACCESS_KEY_ID={self.config.credentials.cloud.aws_access_key_id}
            export AWS_SECRET_ACCESS_KEY={self.config.credentials.cloud.aws_secret_access_key}

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

    def build_database(self):
        """Builds a MySQL central database."""

        db_security_group = resource.aws_security_group(
            "db-security-group",
            name=f"{self.config.app.name}-db-security-group",
            vpc_id=var(self.vpc.id),
            ingress=[
                {
                    "description": "EC2 connection to MySQL database",
                    "from_port": 3306,
                    "to_port": 3306,
                    "protocol": "tcp",
                    "cidr_blocks": [],
                    "ipv6_cidr_blocks": [],
                    "prefix_list_ids": [],
                    "security_groups": [var(self.security_group.id)],
                    "self": False,
                }
            ],
            egress=[
                {
                    "description": "Egress Connection",
                    "from_port": 0,
                    "to_port": 0,
                    "protocol": "-1",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                }
            ],
            tags={"Name": f"pygrid-{self.config.app.name}-db-security-group"},
        )
        self.tfscript += db_security_group

        db_subnet_group = resource.aws_db_subnet_group(
            "default",
            name=f"{self.config.app.name}-db-subnet-group",
            subnet_ids=[var(private_subnet.id) for private_subnet, _ in self.subnets],
            tags={"Name": f"pygrid-{self.config.app.name}-db-subnet-group"},
        )
        self.tfscript += db_subnet_group

        self.database = resource.aws_db_instance(
            f"pygrid-{self.config.app.name}-database",
            engine="mysql",
            port="3306",
            name="pygridDB",
            instance_class="db.t2.micro",
            storage_type="gp2",  # general purpose SSD
            identifier=f"pygrid-{self.config.app.name}-db",  # name
            username=self.config.credentials.db.username,
            password=self.config.credentials.db.password,
            db_subnet_group_name=var(db_subnet_group.id),
            vpc_security_group_ids=[var(db_security_group.id)],
            apply_immediately=True,
            skip_final_snapshot=True,
            # Storage Autoscaling
            allocated_storage=20,
            max_allocated_storage=100,
            tags={"Name": f"pygrid-{self.config.app.name}-mysql-database"},
        )
        self.tfscript += self.database
