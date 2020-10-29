from ...tf import var, var_module
from ..provider import *


class AWS(Provider):
    """Amazon Web Services (AWS) Cloud Provider."""

    def __init__(self, credentials, vpc_config) -> None:
        """
        credentials (dict) : Contains AWS credentials (required for deployment)
        vpc_config (dict) : Contains arguments required to deploy the VPC
        """
        super().__init__()

        credentials_dir = os.path.join(str(Path.home()), ".aws/api/")
        os.makedirs(credentials_dir, exist_ok=True)
        self.credentials = os.path.join(credentials_dir, "credentials.json")

        with open(self.credentials, "w") as cred:
            json.dump(credentials, cred, indent=2, sort_keys=False)

        self.region = vpc_config["region"]
        self.av_zones = vpc_config["av_zones"]

        self.tfscript += terrascript.provider.aws(
            region=self.region, shared_credentials_file=self.credentials
        )

        # Build the VPC
        self.vpc = None
        self.subnets = []
        self.build_vpc()

    # TODO: Make sure this works for serverfull as well.

    def build_vpc(self):
        """
        Appends resources which form the VPC, to the `self.tfscript` configuration object.
        """

        # ----- Virtual Private Cloud (VPC) ------#

        self.vpc = resource.aws_vpc(
            f"pygrid-vpc",
            cidr_block="10.0.0.0/26",  # 2**(32-26) = 64 IP Addresses
            instance_tenancy="default",
            enable_dns_hostnames=True,
            tags={"Name": f"pygrid-vpc"},
        )
        self.tfscript += self.vpc

        # ----- Internet Gateway -----#

        internet_gateway = resource.aws_internet_gateway(
            "igw", vpc_id=var(self.vpc.id), tags={"Name": f"pygrid-igw"}
        )
        self.tfscript += internet_gateway

        # ----- Route Tables -----#

        # One public route table for all public subnets across different availability zones
        public_rt = resource.aws_route_table(
            "public-RT",
            vpc_id=var(self.vpc.id),
            route=[
                {
                    "cidr_block": "0.0.0.0/0",
                    "gateway_id": var(internet_gateway.id),
                    "egress_only_gateway_id": "",
                    "ipv6_cidr_block": "",
                    "instance_id": "",
                    "local_gateway_id": "",
                    "nat_gateway_id": "",
                    "network_interface_id": "",
                    "transit_gateway_id": "",
                    "vpc_peering_connection_id": "",
                }
            ],
            tags={"Name": f"pygrid-public-RT"},
        )
        self.tfscript += public_rt

        # ----- Subnets ----- #

        num_ip_addresses = 2 ** (32 - 26)
        num_subnets = 2 * len(
            self.av_zones
        )  # Each Availability zone contains one public and one private subnet

        cidr_blocks = generate_cidr_block(num_ip_addresses, num_subnets)

        for i, av_zone in enumerate(self.av_zones):
            """
            Each availability zone contains
             - one public subnet : Connects to the internet via public route table
             - one private subnet : Hosts the deployed resources
             - one NAT gateway (in the public subnet) : Allows traffic from the internet to the private subnet
                via the public subnet
             - one Route table : Routes the traffic from the NAT gateway to the private subnet
            """

            private_subnet = resource.aws_subnet(
                f"private-subnet-{i}",
                vpc_id=var(self.vpc.id),
                cidr_block=next(cidr_blocks),
                availability_zone=av_zone,
                tags={"Name": f"private-{i}"},
            )
            self.tfscript += private_subnet

            public_subnet = resource.aws_subnet(
                f"public-subnet-{i}",
                vpc_id=var(self.vpc.id),
                cidr_block=next(cidr_blocks),
                availability_zone=av_zone,
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
                    }
                ],
                tags={"Name": f"pygrid-private-RT-{i}"},
            )
            self.tfscript += private_rt

            # Associate public subnet with public route table
            self.tfscript += resource.aws_route_table_association(
                f"rta-public-subnet-{i}",
                subnet_id=var(public_subnet.id),
                route_table_id=var(public_rt.id),
            )

            # Associate private subnet with private route table
            self.tfscript += resource.aws_route_table_association(
                f"rta-private-subnet-{i}",
                subnet_id=var(private_subnet.id),
                route_table_id=var(private_rt.id),
            )
