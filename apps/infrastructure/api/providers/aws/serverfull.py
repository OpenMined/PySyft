from ...tf import var
from .aws import *


class AWS_Serverfull(AWS):
    def __init__(self, config) -> None:
        """
        credentials (dict) : Contains AWS credentials
        """

        super().__init__(config)

        # Order matters
        self.build_security_group()

        self.build_database()

        # self.writing_exec_script()
        self.build_instance()
        self.build_load_balancer()

        self.output()

    def output(self):
        for count in range(self.config.app.count):
            self.tfscript += terrascript.Output(
                f"instance_{count}_endpoint",
                value=var_module(self.instances[count], "public_ip"),
                description=f"The public IP address of #{count} instance.",
            )

        self.tfscript += terrascript.Output(
            "load_balancer_dns",
            value=var_module(self.load_balancer, "this_elb_dns_name"),
            description="The DNS name of the ELB.",
        )

    def build_security_group(self):
        # ----- Security Group ------#

        self.security_group = resource.aws_security_group(
            "security_group",
            name=f"pygrid-{self.config.app.name}-security-group",
            vpc_id=var(self.vpc.id),
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
            tags={"Name": "pygrid-security-group"},
        )
        self.tfscript += self.security_group

    def build_instance(self):
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
        for count in range(self.config.app.count):
            app = self.config.apps[count]
            instance = Module(
                f"pygrid-instance-{count}",
                instance_count=1,
                source="terraform-aws-modules/ec2-instance/aws",
                name=f"pygrid-{self.config.app.name}-instance-{count}",
                ami=var(self.ami.id),
                instance_type=self.config.vpc.instance_type.split(" ")[1],
                associate_public_ip_address=True,
                monitoring=True,
                vpc_security_group_ids=[var(self.security_group.id)],
                subnet_ids=[var(public_subnet.id) for _, public_subnet in self.subnets],
                # user_data=var(f'file("{self.root_dir}/deploy.sh")'),
                user_data=self.exec_script(app),
                tags={"Name": f"pygrid-{self.config.app.name}-instances-{count}"},
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
            instances=[
                var_module(self.instances[i], f"id[{i}]")
                for i in range(self.config.app.count)
            ],
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

    def exec_script(self, app):
        exec_script = f'''
        #cloud-boothook
        #!/bin/bash

        ## For debugging
        # redirect stdout/stderr to a file
        exec &> log.out


        echo "Simple Web Server for testing the deployment"
        sudo apt update -y
        sudo apt install apache2 -y
        sudo systemctl start apache2
        echo """
        <h1 style='color:#f09764; text-align:center'>
            OpenMined First Server Deployed via Terraform
        </h1>
        """ | sudo tee /var/www/html/index.html

        echo "Setup Miniconda environment"

        sudo wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        sudo bash miniconda.sh -b -p miniconda
        sudo rm miniconda.sh
        export PATH=/miniconda/bin:$PATH > ~/.bashrc
        conda init bash
        source ~/.bashrc
        conda create -y -n pygrid python=3.7
        conda activate pygrid

        echo "Install poetry..."
        pip install poetry

        echo "Install GCC"
        sudo apt-get install python3-dev -y
        sudo apt-get install libevent-dev -y
        sudo apt-get install gcc -y

        echo "Cloning PyGrid"
        git clone https://github.com/OpenMined/PyGrid

        cd /PyGrid/apps/{self.config.app.name}

        echo "Installing {self.config.app.name} Dependencies"
        poetry install

        echo "Setting Database URL"
        export DATABASE_URL={self.database.engine}:pymysql://{self.database.username}:{self.database.password}@{var(self.database.endpoint)}://{self.database.name}

        nohup ./run.sh --port {app.port}  --host {app.host} {f"--id {app.id} --network {app.network}" if self.config.app.name == "domain" else ""}
        '''

        # with open(f"{self.root_dir}/deploy.sh", "w") as deploy_file:
        # deploy_file.write(exec_script)
        return exec_script
