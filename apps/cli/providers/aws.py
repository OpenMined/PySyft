import boto3
import click
from PyInquirer import prompt

from ..deploy import base_setup
from ..tf import *
from ..utils import Config, styles
from .provider import *


## MARK: regions and instances takes some time to be loaded (5-7 sec)
class EC2:
    def __init__(self) -> None:
        self.client = boto3.client("ec2")

    def regions_list(self):
        return [region["RegionName"] for region in self.client.describe_regions()["Regions"]]

    def instances_list(self):
        return [
            instance["InstanceType"]
            for instance in self.client.describe_instance_types()["InstanceTypes"]
        ]


class AWS(Provider):
    """Amazon Web Services (AWS) Cloud Provider."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.config.aws = self.get_aws_config()

        # AWS provider
        self.tfscript += terrascript.provider.aws(
            region=self.config.aws.region,
            shared_credentials_file=self.config.credentials,
        )

        self.update_script()

        click.echo("Initializing AWS Provider")
        TF.init()

        build = self.build()

        if build == 0:
            click.echo("Main Infrastructure has built Successfully!\n\n")

    def build(self) -> bool:
        # Create Virtual Private Cloud (VPC)
        self.main_vpc = terrascript.resource.aws_vpc(
            "main_vpc",
            cidr_block=self.config.aws.vpc_cidr_block,
            instance_tenancy="default",
            enable_dns_hostnames=True,
            tags={"Name": "main-vpc"},
        )
        self.tfscript += self.main_vpc

        # Create Internet Gateway
        self.gw = terrascript.resource.aws_internet_gateway(
            "gw", vpc_id=self.main_vpc.id, tags={"Name": "main-gw"},
        )
        self.tfscript += self.gw

        # Create Route Table
        self.route_table = terrascript.resource.aws_route_table(
            "route_table",
            vpc_id=self.main_vpc.id,
            route=[
                {
                    "cidr_block": "0.0.0.0/0",
                    "gateway_id": self.gw.id,
                    "egress_only_gateway_id": self.gw.id,
                    "ipv6_cidr_block": "::/0",
                    "instance_id": "",
                    "local_gateway_id": "",
                    "nat_gateway_id": "",
                    "network_interface_id": "",
                    "transit_gateway_id": "",
                    "vpc_peering_connection_id": "",
                }
            ],
            tags={"Name": "main-route-table"},
        )
        self.tfscript += self.route_table

        # Create subnet for webservers
        self.main_subnet = terrascript.resource.aws_subnet(
            "main_subnet",
            vpc_id=self.main_vpc.id,
            cidr_block=self.config.aws.subnet_cidr_block,
            tags={"Name": "main-subnet"},
        )
        self.tfscript += self.main_subnet

        # Associate subnet with the route table
        self.rta = terrascript.resource.aws_route_table_association(
            "rta", subnet_id=self.main_subnet.id, route_table_id=self.route_table.id,
        )
        self.tfscript += self.rta

        # Create security group
        self.security_groups = terrascript.resource.aws_security_group(
            "security_groups",
            name="allow_web_traffic",
            description="Allow web inbound traffic",
            vpc_id=self.main_vpc.id,
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
                    "self": True,
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
                    "self": True,
                },
                {
                    "description": "PyGrid Nodes",
                    "from_port": 5000,
                    "to_port": 5999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": True,
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
                    "self": True,
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
                    "self": True,
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
                    "self": True,
                }
            ],
            tags={"Name": "allow_web"},
        )
        self.tfscript += self.security_groups

        self.update_script()
        return TF.validate()

    def deploy_network(
        self, apply: bool = True,
    ):
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
                {"name": "virtualization-type", "values": ["hvm"],},
            ],
            owners=["099720109477"],
        )
        self.tfscript += self.ami

        self.instance = terrascript.resource.aws_instance(
            f"PyGridNetworkInstance",
            ami=self.ami.id,
            instance_type=self.config.aws.instance_type,
            associate_public_ip_address=True,
            vpc_security_group_ids=[self.security_groups.id],
            subnet_id=self.main_subnet.id,
            user_data=f"""
                {base_setup}
                cd /PyGrid/apps/network
                poetry install
                nohup ./run.sh --port {self.config.app.port}  --host {self.config.app.host} {'--start_local_db' if self.config.app.start_local_db else ''}
            """,
        )
        self.tfscript += self.instance

        self.update_script()

        return TF.apply()

    def deploy_node(
        self, apply: bool = True,
    ):
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
                {"name": "virtualization-type", "values": ["hvm"],},
            ],
            owners=["099720109477"],
        )
        self.tfscript += self.ami

        self.instance = terrascript.resource.aws_instance(
            f"PyGridNodeInstance",
            ami=self.ami.id,
            instance_type=self.config.aws.instance_type,
            associate_public_ip_address=True,
            vpc_security_group_ids=[self.security_groups.id],
            subnet_id=self.main_subnet.id,
            user_data=f"""
                {base_setup}
                cd /PyGrid/apps/node
                poetry install
                nohup ./run.sh --id {self.config.app.id} --port {self.config.app.port}  --host {self.config.app.host} --network {self.config.app.network} --num_replicas {self.config.app.num_replicas} {'--start_local_db' if self.config.app.start_local_db else ''}
            """,
        )
        self.tfscript += self.instance

        self.update_script()

        return TF.apply()

    def get_aws_config(self) -> Config:
        """Getting the configration required for deployment on AWs.

        Returns:
            Config: Simple Config with the user inputs
        """
        ec2 = EC2()

        region = prompt(
            [
                {
                    "type": "list",
                    "name": "region",
                    "message": "Please select your desired AWS region",
                    "default": "us-east-1",
                    "choices": ec2.regions_list(),
                },
            ],
            style=styles.second,
        )["region"]

        instance_type = prompt(
            [
                {
                    "type": "list",
                    "name": "instance",
                    "message": "Please select your desired AWS instance type",
                    "default": "t2.micro",
                    "choices": ec2.instances_list(),
                },
            ],
            style=styles.second,
        )["instance"]

        ## VPC
        vpc_cidr_block = prompt(
            [
                {
                    "type": "input",
                    "name": "vpc_cidr_block",
                    "message": "Please provide VPC cidr block",
                    "default": "10.0.0.0/16",
                    # TODO: 'validate': make sure it's a correct ip format
                },
            ],
            style=styles.second,
        )["vpc_cidr_block"]

        ## subnets
        subnet_cidr_block = prompt(
            [
                {
                    "type": "input",
                    "name": "subnet_cidr_block",
                    "message": "Please provide Subnet cidr block",
                    "default": "10.0.0.0/24",
                    # TODO: 'validate': make sure it's a correct ip format
                },
            ],
            style=styles.second,
        )["subnet_cidr_block"]

        return Config(
            region=region,
            instance_type=instance_type,
            vpc_cidr_block=vpc_cidr_block,
            subnet_cidr_block=subnet_cidr_block,
        )
