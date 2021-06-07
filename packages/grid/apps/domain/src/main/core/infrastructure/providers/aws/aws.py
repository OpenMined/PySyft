# grid relative
from ...tf import generate_cidr_block
from ...tf import var
from ...tf import var_module
from ..provider import *


class AWS(Provider):
    """Amazon Web Services (AWS) Cloud Provider."""

    def __init__(self, config: SimpleNamespace) -> None:
        """
        config (SimpleNamespace) : Object storing the required configuration for deployment
        """
        super().__init__(config.root_dir, "aws")
        self.config = config

        # Credentials read from env variables
        self.tfscript += terrascript.provider.aws(region=self.config.vpc.region)

        self.vpc = None
        self.subnets = []

    def build_vpc(self):
        """Adds a VPC."""
        self.vpc = resource.aws_vpc(
            f"pygrid-vpc",
            cidr_block="10.0.0.0/16",
            instance_tenancy="default",
            enable_dns_hostnames=True,
            tags={"Name": f"pygrid-vpc"},
        )
        self.tfscript += self.vpc

    def build_igw(self):
        """Adds an Internet Gateway."""
        self.internet_gateway = resource.aws_internet_gateway(
            "igw", vpc_id=var(self.vpc.id), tags={"Name": f"pygrid-igw"}
        )
        self.tfscript += self.internet_gateway

    def build_public_rt(self):
        """Adds a public Route table.

        One public route table for all public subnets across different
        availability zones
        """
        self.public_rt = resource.aws_route_table(
            "public-RT",
            vpc_id=var(self.vpc.id),
            route=[
                {
                    "cidr_block": "0.0.0.0/0",
                    "gateway_id": var(self.internet_gateway.id),
                    "egress_only_gateway_id": "",
                    "ipv6_cidr_block": "",
                    "instance_id": "",
                    "local_gateway_id": "",
                    "nat_gateway_id": "",
                    "network_interface_id": "",
                    "transit_gateway_id": "",
                    "vpc_peering_connection_id": "",
                    "vpc_endpoint_id": "",
                }
            ],
            tags={"Name": f"pygrid-public-RT"},
        )
        self.tfscript += self.public_rt

    def build_subnets(self):
        """Adds subnets to the VPC. Each availability zone contains.

        - one public subnet : Connects to the internet via public route table
        - one private subnet : Hosts the deployed resources
        - one NAT gateway (in the public subnet) : Allows traffic from the internet to the private subnet
           via the public subnet
        - one Route table : Routes the traffic from the NAT gateway to the private subnet
        """

        for i, av_zone in enumerate(self.config.vpc.av_zones):
            private_subnet = resource.aws_subnet(
                f"private-subnet-{i}",
                vpc_id=var(self.vpc.id),
                cidr_block=generate_cidr_block(
                    base_cidr_block=self.vpc.cidr_block, netnum=(2 * i)
                ),
                availability_zone=av_zone,
                map_public_ip_on_launch=True,
                tags={"Name": f"private-{i}"},
            )
            self.tfscript += private_subnet

            public_subnet = resource.aws_subnet(
                f"public-subnet-{i}",
                vpc_id=var(self.vpc.id),
                cidr_block=generate_cidr_block(
                    base_cidr_block=self.vpc.cidr_block, netnum=(2 * i + 1)
                ),
                availability_zone=av_zone,
                map_public_ip_on_launch=True,
                tags={"Name": f"public-{i}"},
            )
            self.tfscript += public_subnet

            self.subnets.append((private_subnet, public_subnet))

            # Elastic IP for NAT Gateway
            elastic_ip = resource.aws_eip(
                f"eip-{i}", vpc=True, tags={"Name": f"pygrid-EIP-{i}"}
            )
            self.tfscript += elastic_ip

            # NAT Gateway
            nat_gateway = resource.aws_nat_gateway(
                f"ngw-{i}",
                allocation_id=var(elastic_ip.id),
                subnet_id=var(public_subnet.id),
                tags={"Name": f"pygrid-ngw-{i}"},
            )
            self.tfscript += nat_gateway

            # Route table for private subnet
            private_rt = resource.aws_route_table(
                f"private-RT-{i}",
                vpc_id=var(self.vpc.id),
                route=[
                    {
                        "cidr_block": "0.0.0.0/0",
                        "nat_gateway_id": var(nat_gateway.id),
                        "ipv6_cidr_block": "",
                        "gateway_id": "",
                        "egress_only_gateway_id": "",
                        "instance_id": "",
                        "local_gateway_id": "",
                        "network_interface_id": "",
                        "transit_gateway_id": "",
                        "vpc_peering_connection_id": "",
                        "vpc_endpoint_id": "",
                    }
                ],
                tags={"Name": f"pygrid-private-RT-{i}"},
            )
            self.tfscript += private_rt

            # Associate public subnet with public route table
            self.tfscript += resource.aws_route_table_association(
                f"rta-public-subnet-{i}",
                subnet_id=var(public_subnet.id),
                route_table_id=var(self.public_rt.id),
            )

            # Associate private subnet with private route table
            self.tfscript += resource.aws_route_table_association(
                f"rta-private-subnet-{i}",
                subnet_id=var(private_subnet.id),
                route_table_id=var(private_rt.id),
            )
