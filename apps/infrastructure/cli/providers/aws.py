import boto3
import click
from PyInquirer import prompt

from ..utils import Config, styles


def get_vpc_config() -> Config:
    """Assists the user in entering configuration related to VPC.

    :return:
    """

    region = prompt(
        [
            {
                "type": "list",
                "name": "region",
                "message": "Please select your desired AWS region",
                "default": "us-east-1",
                "choices": [
                    region["RegionName"]
                    for region in boto3.client("ec2").describe_regions()["Regions"]
                ],
            }
        ],
        style=styles.second,
    )["region"]

    av_zones = prompt(
        [
            {
                "type": "checkbox",
                "name": "av_zones",
                "message": "Please select atleast two availability zones. (Not sure? Select the first two)",
                "choices": [
                    {"name": zone["ZoneName"]}
                    for zone in boto3.client(
                        "ec2", region_name=region
                    ).describe_availability_zones(
                        Filters=[{"Name": "region-name", "Values": [region]}]
                    )[
                        "AvailabilityZones"
                    ]
                ],
            }
        ],
        style=styles.second,
    )["av_zones"]

    return Config(region=region, av_zones=av_zones)


def get_instance_type():
    instance_info = (
        lambda i: f"Instance: {i['InstanceType']} # Memory: {round(i['MemoryInfo']['SizeInMiB'] / 1024, 3)}GB # CPUs: {i['VCpuInfo']['DefaultVCpus']}"
    )
    instances = sorted(
        boto3.client("ec2").describe_instance_types()["InstanceTypes"],
        key=lambda i: i["InstanceType"],
    )
    return prompt(
        [
            {
                "type": "list",
                "name": "instance",
                "message": "Please select your desired AWS instance",
                "default": "t2.micro",
                "choices": [instance_info(instance) for instance in instances],
            }
        ],
        style=styles.second,
    )["instance"]


def get_vpc_ip_config() -> Config:
    """Assists the user in entering configuration related to IP address of VPC.

    :return:
    """

    cidr_blocks = prompt(
        [
            {
                "type": "input",
                "name": "vpc_cidr_block",
                "message": "Please provide VPC cidr block",
                "default": "10.0.0.0/16",
            },
            {
                "type": "input",
                "name": "subnet_cidr_block",
                "message": "Please provide Subnet cidr block",
                "default": "10.0.0.0/24",
            },
        ],
        style=styles.second,
    )

    return Config(
        vpc_cidr_block=cidr_blocks["vpc_cidr_block"],
        subnet_cidr_block=cidr_blocks["subnet_cidr_block"],
    )


def get_db_config() -> Config:
    """Assists the user in entering configuration related the database.

    :return:
    """
    username = prompt(
        [
            {
                "type": "input",
                "name": "username",
                "message": "Please set a username for your Database",
                "validate": lambda x: True
                if len(x) > 4
                else "Username length should be at least 4 characters",
            }
        ],
        style=styles.second,
    )["username"]

    def get_password(msg, validate):
        return prompt(
            [
                {
                    "type": "password",
                    "name": "password",
                    "message": msg,
                    "validate": validate,
                }
            ],
            style=styles.second,
        )["password"]

    password = get_password(
        msg="Enter a password for your Database (length > 8)",
        validate=lambda x: True
        if len(x) > 8
        else "Password length should be greater than 8 characters",
    )
    re_password = get_password(
        msg="Enter the password again",
        validate=lambda x: True
        if x == password
        else "The passwords do not match. Please enter again",
    )

    return Config(username=username, password=password)
