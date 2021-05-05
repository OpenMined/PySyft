# stdlib
import textwrap

# grid relative
from .aws import *


class AWS_Serverfull(AWS):
    def __init__(self, config: SimpleNamespace) -> None:
        """
        credentials (dict) : Contains AWS credentials
        """

        super().__init__(config)

        self.worker = config.app.name == "worker"

        if self.worker:
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
                value=var_module(self.load_balancer, "this_elb_dns_name"),
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
                instance_name = (
                    f"pygrid-{self.config.app.name}-{str(self.config.app.id)}"
                )
                kwargs = {
                    "name": instance_name,
                    "subnet_ids": [
                        public_subnet.id for _, public_subnet in self.subnets
                    ],
                    "tags": {"Name": instance_name},
                }
            else:
                instance_name = f"pygrid-{self.config.app.name}-instance-{count}"
                self.write_exec_script(app, index=count)
                kwargs = {
                    "name": instance_name,
                    "subnet_ids": [
                        var(public_subnet.id) for _, public_subnet in self.subnets
                    ],
                    "user_data": self.write_exec_script(app, index=count),
                    "tags": {"Name": instance_name},
                }

            instance = Module(
                f"pygrid-instance-{count}",
                instance_count=1,
                source="terraform-aws-modules/ec2-instance/aws",
                ami=var(self.ami.id),
                instance_type=self.config.vpc.instance_type.InstanceType,
                associate_public_ip_address=True,
                monitoring=True,
                vpc_security_group_ids=[var(self.security_group.id)],
                **kwargs,
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

    def write_exec_script(self, app, index=0):
        ##TODO(amr): remove `git checkout pygrid_0.3.0` after merge

        # exec_script = "#cloud-boothook\n#!/bin/bash\n"
        exec_script = "#!/bin/bash\n"
        exec_script += textwrap.dedent(
            f"""
            ## For debugging
            # redirect stdout/stderr to a file
            exec &> server_log.out
            echo 'Simple Web Server for testing the deployment'
            sudo apt update -y
            sudo apt install apache2 -y
            sudo systemctl start apache2
            echo '<h1>OpenMined {self.config.app.name} Server ({index}) Deployed via Terraform</h1>' | sudo tee /var/www/html/index.html

            exec &> conda_log.out
            echo 'Setup Miniconda environment'
            sudo wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
            sudo bash miniconda.sh -b -p miniconda
            sudo rm miniconda.sh
            export PATH=/miniconda/bin:$PATH > ~/.bashrc
            conda init bash
            source ~/.bashrc
            conda create -y -n pygrid python=3.7
            conda activate pygrid

            exec &> poetry_log.out
            echo 'Install poetry...'
            pip install poetry

            exec &> gcc_log.out
            echo 'Install GCC'
            sudo apt-get install python3-dev -y
            sudo apt-get install libevent-dev -y
            sudo apt-get install gcc -y

            exec &> grid_log.out
            echo 'Cloning PyGrid'
            git clone https://github.com/OpenMined/PyGrid && cd /PyGrid/
            git checkout infra_workers_0.3

            cd /PyGrid/apps/{self.config.app.name}

            exec &> dependencies_log.out
            echo 'Installing {self.config.app.name} Dependencies'
            poetry install

            ## TODO(amr): remove this after poetry updates
            pip install pymysql

            exec &> start_app.out
            nohup ./run.sh --port {app.port}  --host {app.host}
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
